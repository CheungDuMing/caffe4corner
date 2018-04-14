#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "caffe/layers/detection_pbox_evaluate_layer.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
void DetectionPboxEvaluateLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const DetectionPboxEvaluateParameter& detection_pbox_evaluate_param =
      this->layer_param_.detection_pbox_evaluate_param();
  CHECK(detection_pbox_evaluate_param.has_num_classes())
      << "Must provide num_classes.";
  num_classes_ = detection_pbox_evaluate_param.num_classes();
  background_label_id_ = detection_pbox_evaluate_param.background_label_id();
  overlap_threshold_ = detection_pbox_evaluate_param.overlap_threshold();
  CHECK_LT(overlap_threshold_, 0.) << "overlap_threshold must be negative.";
  evaluate_difficult_gt_ = detection_pbox_evaluate_param.evaluate_difficult_gt();
  if (detection_pbox_evaluate_param.has_name_size_file()) {
    string name_size_file = detection_pbox_evaluate_param.name_size_file();
    std::ifstream infile(name_size_file.c_str());
    CHECK(infile.good())
        << "Failed to open name size file: " << name_size_file;
    // The file is in the following format:
    //    name height width
    //    ...
    string name;
    int height, width;
    while (infile >> name >> height >> width) {
      sizes_.push_back(std::make_pair(height, width));
    }
    infile.close();
  }
  count_ = 0;
  // If there is no name_size_file provided, use normalized bbox to evaluate.
  use_normalized_pbox_ = sizes_.size() == 0;

  // Retrieve resize parameter if there is any provided.
  has_resize_ = detection_pbox_evaluate_param.has_resize_param();
  if (has_resize_) {
    resize_param_ = detection_pbox_evaluate_param.resize_param();
  }
}

template <typename Dtype>
void DetectionPboxEvaluateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(count_, sizes_.size());
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->width(), 11);
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->width(), 12);

  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  int num_pos_classes = background_label_id_ == -1 ?
      num_classes_ : num_classes_ - 1;
  int num_valid_det = 0;
  const Dtype* det_data = bottom[0]->cpu_data();
  for (int i = 0; i < bottom[0]->height(); ++i) {
    if (det_data[1] != -1) {
      ++num_valid_det;
    }
    det_data += 11;
  }
  top_shape.push_back(num_pos_classes + num_valid_det);
  // Each row is a 5 dimension vector, which stores
  // [image_id, label, confidence, true_pos, false_pos]
  top_shape.push_back(5);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DetectionPboxEvaluateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* det_data = bottom[0]->cpu_data();
  const Dtype* gt_data = bottom[1]->cpu_data();

  // Retrieve all detection results.
  map<int, LabelPBox> all_detections;
  GetPboxDetectionResults(det_data, bottom[0]->height(), background_label_id_,
                      &all_detections);

  // Retrieve all ground truth (including difficult ones).
  map<int, LabelPBox> all_gt_pboxes;
  GetPboxGroundTruth(gt_data, bottom[1]->height(), background_label_id_,
                 true, &all_gt_pboxes);

  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0.), top_data);
  int num_det = 0;

  // Insert number of ground truth for each label.
  map<int, int> num_pos;
  for (map<int, LabelPBox>::iterator it = all_gt_pboxes.begin();
       it != all_gt_pboxes.end(); ++it) {
    for (LabelPBox::iterator iit = it->second.begin(); iit != it->second.end();
         ++iit) {
      int count = 0;
      if (evaluate_difficult_gt_) {
        count = iit->second.size();
      } else {
        // Get number of non difficult ground truth.
        for (int i = 0; i < iit->second.size(); ++i) {
          if (!iit->second[i].difficult()) {
            ++count;
          }
        }
      }
      if (num_pos.find(iit->first) == num_pos.end()) {
        num_pos[iit->first] = count;
      } else {
        num_pos[iit->first] += count;
      }
    }
  }
  for (int c = 0; c < num_classes_; ++c) {
    if (c == background_label_id_) {
      continue;
    }
    top_data[num_det * 5] = -1;
    top_data[num_det * 5 + 1] = c;
    if (num_pos.find(c) == num_pos.end()) {
      top_data[num_det * 5 + 2] = 0;
    } else {
      top_data[num_det * 5 + 2] = num_pos.find(c)->second;
    }
    top_data[num_det * 5 + 3] = -1;
    top_data[num_det * 5 + 4] = -1;
    ++num_det;
  }

  // Insert detection evaluate status.
  for (map<int, LabelPBox>::iterator it = all_detections.begin();
       it != all_detections.end(); ++it) {
    int image_id = it->first;
    LabelPBox& detections = it->second;
    if (all_gt_pboxes.find(image_id) == all_gt_pboxes.end()) {
      // No ground truth for current image. All detections become false_pos.
      for (LabelPBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        if (label == -1) {
          continue;
        }
        const vector<NormalizedPBox>& pboxes = iit->second;
        for (int i = 0; i < pboxes.size(); ++i) {
          top_data[num_det * 5] = image_id;
          top_data[num_det * 5 + 1] = label;
          top_data[num_det * 5 + 2] = pboxes[i].score();
          top_data[num_det * 5 + 3] = 0;
          top_data[num_det * 5 + 4] = 1;
          ++num_det;
        }
      }
    } else {
      LabelPBox& label_pboxes = all_gt_pboxes.find(image_id)->second;
      for (LabelPBox::iterator iit = detections.begin();
           iit != detections.end(); ++iit) {
        int label = iit->first;
        if (label == -1) {
          continue;
        }
        vector<NormalizedPBox>& pboxes = iit->second;
        if (label_pboxes.find(label) == label_pboxes.end()) {
          // No ground truth for current label. All detections become false_pos.
          for (int i = 0; i < pboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = pboxes[i].score();
            top_data[num_det * 5 + 3] = 0;
            top_data[num_det * 5 + 4] = 1;
            ++num_det;
          }
        } else {
          vector<NormalizedPBox>& gt_pboxes = label_pboxes.find(label)->second;
          // Scale ground truth if needed.
          if (!use_normalized_pbox_) {
            CHECK_LT(count_, sizes_.size());
            for (int i = 0; i < gt_pboxes.size(); ++i) {
              OutputPBox(gt_pboxes[i], sizes_[count_], has_resize_,
                         resize_param_, &(gt_pboxes[i]));
            }
          }
          vector<bool> visited(gt_pboxes.size(), false);
          // Sort detections in descend order based on scores.
          std::sort(pboxes.begin(), pboxes.end(), SortPBoxDescend);
          for (int i = 0; i < pboxes.size(); ++i) {
            top_data[num_det * 5] = image_id;
            top_data[num_det * 5 + 1] = label;
            top_data[num_det * 5 + 2] = pboxes[i].score();
            if (!use_normalized_pbox_) {
              OutputPBox(pboxes[i], sizes_[count_], has_resize_,
                         resize_param_, &(pboxes[i]));
            }
            // Compare with each ground truth bbox.
            float overlap_max = -1000;
            int jmax = -1;
            for (int j = 0; j < gt_pboxes.size(); ++j) {
              float overlap = JaccardOverlapPbox(pboxes[i], gt_pboxes[j],
                                             use_normalized_pbox_);
             /* LOG(INFO) << pboxes[i].ltopx() << " " << pboxes[i].ltopy();*/
              //LOG(INFO) << pboxes[i].lbottomx() << " " << pboxes[i].lbottomy();
              //LOG(INFO) << pboxes[i].rbottomx() << " " << pboxes[i].rbottomy();
              /*LOG(INFO) << pboxes[i].rtopx() << " " << pboxes[i].rtopy();*/
/*              if (PBoxSize(pboxes[i]) > 0 && PBoxSize(gt_pboxes[i])>0.1) {*/
              //LOG(INFO)<< overlap;
              //LOG(INFO) << "pboxsize: " << PBoxSize(pboxes[i]);
              //LOG(INFO) << "gt_pboxsize: " << PBoxSize(gt_pboxes[i]);

              /*}*/
              //if(overlap > 0) LOG(FATAL) << "success";
              if (overlap > overlap_max) {
                overlap_max = overlap;
                jmax = j;
              }
            }
               //if(overlap_max > -1000) LOG(INFO) << overlap_max;
            if (overlap_max >= overlap_threshold_) {
                LOG(INFO)<< "success!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
              if (evaluate_difficult_gt_ ||
                  (!evaluate_difficult_gt_ && !gt_pboxes[jmax].difficult())) {
                if (!visited[jmax]) {
                  // true positive.
                  top_data[num_det * 5 + 3] = 1;
                  top_data[num_det * 5 + 4] = 0;
                  visited[jmax] = true;
                } else {
                  // false positive (multiple detection).
                  top_data[num_det * 5 + 3] = 0;
                  top_data[num_det * 5 + 4] = 1;
                }
              }
            } else {
              // false positive.
              top_data[num_det * 5 + 3] = 0;
              top_data[num_det * 5 + 4] = 1;
            }
            ++num_det;
          }
        }
      }
    }
    if (sizes_.size() > 0) {
      ++count_;
      if (count_ == sizes_.size()) {
        // reset count after a full iterations through the DB.
        count_ = 0;
      }
    }
  }
}

INSTANTIATE_CLASS(DetectionPboxEvaluateLayer);
REGISTER_LAYER_CLASS(DetectionPboxEvaluate);

}  // namespace caffe
