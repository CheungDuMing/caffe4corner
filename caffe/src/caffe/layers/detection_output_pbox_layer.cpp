#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_pbox_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void DetectionOutputPboxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const DetectionOutputPboxParameter& detection_output_pbox_param =
			this->layer_param_.detection_output_pbox_param();
		CHECK(detection_output_pbox_param.has_num_classes()) << "Must specify num_classes";
		num_classes_ = detection_output_pbox_param.num_classes();
		share_location_ = detection_output_pbox_param.share_location();
		num_loc_classes_ = share_location_ ? 1 : num_classes_;
		background_label_id_ = detection_output_pbox_param.background_label_id();
		code_type_ = detection_output_pbox_param.code_type();
		variance_encoded_in_target_ =
			detection_output_pbox_param.variance_encoded_in_target();
		keep_top_k_ = detection_output_pbox_param.keep_top_k();
		confidence_threshold_ = detection_output_pbox_param.has_confidence_threshold() ?
			detection_output_pbox_param.confidence_threshold() : -FLT_MAX;
		// Parameters used in nms.
		nms_threshold_ = detection_output_pbox_param.nms_param().nms_threshold();
		CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
		eta_ = detection_output_pbox_param.nms_param().eta();
		CHECK_GT(eta_, 0.);
		CHECK_LE(eta_, 1.);
		top_k_ = -1;
		if (detection_output_pbox_param.nms_param().has_top_k()) {
			top_k_ = detection_output_pbox_param.nms_param().top_k();
		}
		const SaveOutputParameter& save_output_param =
			detection_output_pbox_param.save_output_param();
		output_directory_ = save_output_param.output_directory();
		if (!output_directory_.empty()) {
			if (boost::filesystem::is_directory(output_directory_)) {
				boost::filesystem::remove_all(output_directory_);
			}
			if (!boost::filesystem::create_directories(output_directory_)) {
				LOG(WARNING) << "Failed to create directory: " << output_directory_;
			}
		}
		output_name_prefix_ = save_output_param.output_name_prefix();
		need_save_ = output_directory_ == "" ? false : true;
		output_format_ = save_output_param.output_format();
		if (save_output_param.has_label_map_file()) {
			string label_map_file = save_output_param.label_map_file();
			if (label_map_file.empty()) {
				// Ignore saving if there is no label_map_file provided.
				LOG(WARNING) << "Provide label_map_file if output results to files.";
				need_save_ = false;
			}
			else {
				LabelMap label_map;
				CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
					<< "Failed to read label map file: " << label_map_file;
				CHECK(MapLabelToName(label_map, true, &label_to_name_))
					<< "Failed to convert label to name.";
				CHECK(MapLabelToDisplayName(label_map, true, &label_to_display_name_))
					<< "Failed to convert label to display name.";
			}
		}
		else {
			need_save_ = false;
		}
		if (save_output_param.has_name_size_file()) {
			string name_size_file = save_output_param.name_size_file();
			if (name_size_file.empty()) {
				// Ignore saving if there is no name_size_file provided.
				LOG(WARNING) << "Provide name_size_file if output results to files.";
				need_save_ = false;
			}
			else {
				std::ifstream infile(name_size_file.c_str());
				CHECK(infile.good())
					<< "Failed to open name size file: " << name_size_file;
				// The file is in the following format:
				//    name height width
				//    ...
				string name;
				int height, width;
				while (infile >> name >> height >> width) {
					names_.push_back(name);
					sizes_.push_back(std::make_pair(height, width));
				}
				infile.close();
				if (save_output_param.has_num_test_image()) {
					num_test_image_ = save_output_param.num_test_image();
				}
				else {
					num_test_image_ = names_.size();
				}
				CHECK_LE(num_test_image_, names_.size());
			}
		}
		else {
			need_save_ = false;
		}
		has_resize_ = save_output_param.has_resize_param();
		if (has_resize_) {
			resize_param_ = save_output_param.resize_param();
		}
		name_count_ = 0;
		visualize_ = detection_output_pbox_param.visualize();
		if (visualize_) {
			visualize_threshold_ = 0.6;
			if (detection_output_pbox_param.has_visualize_threshold()) {
				visualize_threshold_ = detection_output_pbox_param.visualize_threshold();
			}
			data_transformer_.reset(
				new DataTransformer<Dtype>(this->layer_param_.transform_param(),
					this->phase_));
			data_transformer_->InitRand();
			save_file_ = detection_output_pbox_param.save_file();
		}
		pbox_preds_.ReshapeLike(*(bottom[0]));
		if (!share_location_) {
			pbox_permute_.ReshapeLike(*(bottom[0]));
		}
		conf_permute_.ReshapeLike(*(bottom[1]));
	}

	template <typename Dtype>
	void DetectionOutputPboxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (need_save_) {
			CHECK_LE(name_count_, names_.size());
			if (name_count_ % num_test_image_ == 0) {
				// Clean all outputs.
				if (output_format_ == "VOC") {
					boost::filesystem::path output_directory(output_directory_);
					for (map<int, string>::iterator it = label_to_name_.begin();
						it != label_to_name_.end(); ++it) {
						if (it->first == background_label_id_) {
							continue;
						}
						std::ofstream outfile;
						boost::filesystem::path file(
							output_name_prefix_ + it->second + ".txt");
						boost::filesystem::path out_file = output_directory / file;
						outfile.open(out_file.string().c_str(), std::ofstream::out);
					}
				}
			}
		}
		CHECK_EQ(bottom[0]->num(), bottom[1]->num());
		if (pbox_preds_.num() != bottom[0]->num() ||
			pbox_preds_.count(1) != bottom[0]->count(1)) {
			pbox_preds_.ReshapeLike(*(bottom[0]));
		}
		if (!share_location_ && (pbox_permute_.num() != bottom[0]->num() ||
			pbox_permute_.count(1) != bottom[0]->count(1))) {
			pbox_permute_.ReshapeLike(*(bottom[0]));
		}
		if (conf_permute_.num() != bottom[1]->num() ||
			conf_permute_.count(1) != bottom[1]->count(1)) {
			conf_permute_.ReshapeLike(*(bottom[1]));
		}
		num_priors_ = bottom[2]->height() / 8;
		CHECK_EQ(num_priors_ * num_loc_classes_ * 8, bottom[0]->channels())
			<< "Number of priors must match number of location predictions.";
		CHECK_EQ(num_priors_ * num_classes_ , bottom[1]->channels())
			<< "Number of priors must match number of confidence predictions.";
		// num() and channels() are 1.
		vector<int> top_shape(2, 1);
		// Since the number of pboxes to be kept is unknown before nms, we manually
		// set it to (fake) 1.
		top_shape.push_back(1);
		// Each row is a 11 dimension vector, which stores
		// [image_id, label, confidence, ltopx, ...., rtopy]
		top_shape.push_back(11);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DetectionOutputPboxLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LOG(FATAL) << "detectionOutputPbox.cpp success";
		const Dtype* loc_data = bottom[0]->cpu_data();
		const Dtype* conf_data = bottom[1]->cpu_data();
		const Dtype* prior_data = bottom[2]->cpu_data();
		const int num = bottom[0]->num();

		// Retrieve all location predictions.
		vector<LabelPBox> all_loc_preds;
		GetPboxLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
			share_location_, &all_loc_preds);

		// Retrieve all confidences.
		vector<map<int, vector<float> > > all_conf_scores;
		GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
			&all_conf_scores);

		// Retrieve all prior bboxes. It is same within a batch since we assume all
		// images in a batch are of same dimension.
		vector<NormalizedPBox> prior_pboxes;
		vector<vector<float> > prior_variances;
		GetPriorPBoxes(prior_data, num_priors_, &prior_pboxes, &prior_variances);

		// Decode all loc predictions to bboxes.
		vector<LabelPBox> all_decode_pboxes;
		const bool clip_pbox = false;
		//********************************************************************
		DecodePBoxesAll(all_loc_preds, prior_pboxes, prior_variances, num,
			share_location_, num_loc_classes_, background_label_id_,
			code_type_, variance_encoded_in_target_, clip_pbox,
			&all_decode_pboxes);

		int num_kept = 0;
		vector<map<int, vector<int> > > all_indices;
		for (int i = 0; i < num; ++i) {
			const LabelPBox& decode_pboxes = all_decode_pboxes[i];
			const map<int, vector<float> >& conf_scores = all_conf_scores[i];
			map<int, vector<int> > indices;
			int num_det = 0;
			for (int c = 0; c < num_classes_; ++c) {
				if (c == background_label_id_) {
					// Ignore background class.
					continue;
				}
				if (conf_scores.find(c) == conf_scores.end()) {
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find confidence predictions for label " << c;
				}
				const vector<float>& scores = conf_scores.find(c)->second;
				int label = share_location_ ? -1 : c;
				if (decode_pboxes.find(label) == decode_pboxes.end()) {
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find location predictions for label " << label;
					continue;
				}
				const vector<NormalizedPBox>& pboxes = decode_pboxes.find(label)->second;
				ApplyNMSFast(pboxes, scores, confidence_threshold_, nms_threshold_, eta_,
					top_k_, &(indices[c]));
				num_det += indices[c].size();
			}
			if (keep_top_k_ > -1 && num_det > keep_top_k_) {
				vector<pair<float, pair<int, int> > > score_index_pairs;
				for (map<int, vector<int> >::iterator it = indices.begin();
					it != indices.end(); ++it) {
					int label = it->first;
					const vector<int>& label_indices = it->second;
					if (conf_scores.find(label) == conf_scores.end()) {
						// Something bad happened for current label.
						LOG(FATAL) << "Could not find location predictions for " << label;
						continue;
					}
					const vector<float>& scores = conf_scores.find(label)->second;
					for (int j = 0; j < label_indices.size(); ++j) {
						int idx = label_indices[j];
						CHECK_LT(idx, scores.size());
						score_index_pairs.push_back(std::make_pair(
							scores[idx], std::make_pair(label, idx)));
					}
				}
				// Keep top k results per image.
				std::sort(score_index_pairs.begin(), score_index_pairs.end(),
					SortScorePairDescend<pair<int, int> >);
				score_index_pairs.resize(keep_top_k_);
				// Store the new indices.
				map<int, vector<int> > new_indices;
				for (int j = 0; j < score_index_pairs.size(); ++j) {
					int label = score_index_pairs[j].second.first;
					int idx = score_index_pairs[j].second.second;
					new_indices[label].push_back(idx);
				}
				all_indices.push_back(new_indices);
				num_kept += keep_top_k_;
			}
			else {
				all_indices.push_back(indices);
				num_kept += num_det;
			}
		}

		vector<int> top_shape(2, 1);
		top_shape.push_back(num_kept);
		top_shape.push_back(11);
		Dtype* top_data;
		if (num_kept == 0) {
			LOG(INFO) << "Couldn't find any detections";
			top_shape[2] = num;
			top[0]->Reshape(top_shape);
			top_data = top[0]->mutable_cpu_data();
			caffe_set<Dtype>(top[0]->count(), -1, top_data);
			// Generate fake results per image.
			for (int i = 0; i < num; ++i) {
				top_data[0] = i;
				top_data += 11;
			}
		}
		else {
			top[0]->Reshape(top_shape);
			top_data = top[0]->mutable_cpu_data();
		}

		int count = 0;
		boost::filesystem::path output_directory(output_directory_);
		for (int i = 0; i < num; ++i) {
			const map<int, vector<float> >& conf_scores = all_conf_scores[i];
			const LabelPBox& decode_pboxes = all_decode_pboxes[i];
			for (map<int, vector<int> >::iterator it = all_indices[i].begin();
				it != all_indices[i].end(); ++it) {
				int label = it->first;
				if (conf_scores.find(label) == conf_scores.end()) {
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find confidence predictions for " << label;
					continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				int loc_label = share_location_ ? -1 : label;
				if (decode_pboxes.find(loc_label) == decode_pboxes.end()) {
					// Something bad happened if there are no predictions for current label.
					LOG(FATAL) << "Could not find location predictions for " << loc_label;
					continue;
				}
				const vector<NormalizedPBox>& pboxes =
					decode_pboxes.find(loc_label)->second;
				vector<int>& indices = it->second;
				if (need_save_) {
					CHECK(label_to_name_.find(label) != label_to_name_.end())
						<< "Cannot find label: " << label << " in the label map.";
					CHECK_LT(name_count_, names_.size());
				}
				for (int j = 0; j < indices.size(); ++j) {
					int idx = indices[j];
					top_data[count * 11] = i;
					top_data[count * 11 + 1] = label;
					top_data[count * 11 + 2] = scores[idx];
					const NormalizedPBox& pbox = pboxes[idx];
					top_data[count * 11 + 3] = pbox.ltopx();
					top_data[count * 11 + 4] = pbox.ltopy();
					top_data[count * 11 + 5] = pbox.lbottomx();
					top_data[count * 11 + 6] = pbox.lbottomy();
					top_data[count * 11 + 7] = pbox.rbottomx();
					top_data[count * 11 + 8] = pbox.rbottomy();
					top_data[count * 11 + 9] = pbox.rtopx();
					top_data[count * 11 + 10] = pbox.rtopy();
					if (need_save_) {
						NormalizedPBox out_pbox;
						//*******************************************************************
						OutputPBox(pbox, sizes_[name_count_], has_resize_, resize_param_,
							&out_pbox);
						float score = top_data[count * 11 + 2];
						float ltopx = out_pbox.ltopx();
						float ltopy = out_pbox.ltopy();
						float lbottomx = out_pbox.lbottomx();
						float lbottomy = out_pbox.lbottomy();
						float rbottomx = out_pbox.rbottomx();
						float rbottomy = out_pbox.rbottomy();
						float rtopx = out_pbox.rtopx();
						float rtopy = out_pbox.rtopy();

						ptree pt_ltopx, pt_ltopy, pt_lbottomx, pt_lbottomy,
							pt_rbottomx, pt_rbottomy, pt_rtopx, pt_rtopy;
						pt_ltopx.put<float>("", round(ltopx * 100) / 100.);
						pt_ltopy.put<float>("", round(ltopy * 100) / 100.);
						pt_lbottomx.put<float>("", round(lbottomx * 100) / 100.);
						pt_lbottomy.put<float>("", round(lbottomy * 100) / 100.);
						pt_rbottomx.put<float>("", round(rbottomx * 100) / 100.);
						pt_rbottomy.put<float>("", round(rbottomy * 100) / 100.);
						pt_rtopx.put<float>("", round(rtopx * 100) / 100.);
						pt_rtopy.put<float>("", round(rtopy * 100) / 100.);

						ptree cur_pbox;
						cur_pbox.push_back(std::make_pair("", pt_ltopx));
						cur_pbox.push_back(std::make_pair("", pt_ltopy));
						cur_pbox.push_back(std::make_pair("", pt_lbottomx));
						cur_pbox.push_back(std::make_pair("", pt_lbottomy));
						cur_pbox.push_back(std::make_pair("", pt_rbottomx));
						cur_pbox.push_back(std::make_pair("", pt_rbottomy));
						cur_pbox.push_back(std::make_pair("", pt_rtopx));
						cur_pbox.push_back(std::make_pair("", pt_rtopy));

						ptree cur_det;
						cur_det.put("image_id", names_[name_count_]);
						if (output_format_ == "ILSVRC") {
							cur_det.put<int>("category_id", label);
						}
						else {
							cur_det.put("category_id", label_to_name_[label].c_str());
						}
						cur_det.add_child("pbox", cur_pbox);
						cur_det.put<float>("score", score);

						detections_.push_back(std::make_pair("", cur_det));
					}
					++count;
				}
			}
			if (need_save_) {
				++name_count_;
				if (name_count_ % num_test_image_ == 0) {
					if (output_format_ == "VOC") {
						map<string, std::ofstream*> outfiles;
						for (int c = 0; c < num_classes_; ++c) {
							if (c == background_label_id_) {
								continue;
							}
							string label_name = label_to_name_[c];
							boost::filesystem::path file(
								output_name_prefix_ + label_name + ".txt");
							boost::filesystem::path out_file = output_directory / file;
							outfiles[label_name] = new std::ofstream(out_file.string().c_str(),
								std::ofstream::out);
						}
						BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
							ptree pt = det.second;
							string label_name = pt.get<string>("category_id");
							if (outfiles.find(label_name) == outfiles.end()) {
								std::cout << "Cannot find " << label_name << std::endl;
								continue;
							}
							string image_name = pt.get<string>("image_id");
							float score = pt.get<float>("score");
							//**********************************************************
							vector<int> pbox;
							BOOST_FOREACH(ptree::value_type &elem, pt.get_child("pbox")) {
								pbox.push_back(static_cast<int>(elem.second.get_value<float>()));
							}
							*(outfiles[label_name]) << image_name;
							*(outfiles[label_name]) << " " << score;
							*(outfiles[label_name]) << " " << pbox[0] << " " << pbox[1];
							*(outfiles[label_name]) << " " << pbox[2] << " " << pbox[3];
							*(outfiles[label_name]) << " " << pbox[4] << " " << pbox[5];
							*(outfiles[label_name]) << " " << pbox[6] << " " << pbox[7];
							LOG(FATAL) << "import successfully " << pbox[0];
							*(outfiles[label_name]) << std::endl;
						}
						for (int c = 0; c < num_classes_; ++c) {
							if (c == background_label_id_) {
								continue;
							}
							string label_name = label_to_name_[c];
							outfiles[label_name]->flush();
							outfiles[label_name]->close();
							delete outfiles[label_name];
						}
					}
					else if (output_format_ == "COCO") {
						boost::filesystem::path output_directory(output_directory_);
						boost::filesystem::path file(output_name_prefix_ + ".json");
						boost::filesystem::path out_file = output_directory / file;
						std::ofstream outfile;
						outfile.open(out_file.string().c_str(), std::ofstream::out);

						boost::regex exp("\"(null|true|false|-?[0-9]+(\\.[0-9]+)?)\"");
						ptree output;
						output.add_child("detections", detections_);
						std::stringstream ss;
						write_json(ss, output);
						std::string rv = boost::regex_replace(ss.str(), exp, "$1");
						outfile << rv.substr(rv.find("["), rv.rfind("]") - rv.find("["))
							<< std::endl << "]" << std::endl;
					}
					else if (output_format_ == "ILSVRC") {
						boost::filesystem::path output_directory(output_directory_);
						boost::filesystem::path file(output_name_prefix_ + ".txt");
						boost::filesystem::path out_file = output_directory / file;
						std::ofstream outfile;
						outfile.open(out_file.string().c_str(), std::ofstream::out);

						BOOST_FOREACH(ptree::value_type &det, detections_.get_child("")) {
							ptree pt = det.second;
							int label = pt.get<int>("category_id");
							string image_name = pt.get<string>("image_id");
							float score = pt.get<float>("score");
							//*************************************************************
							vector<int> pbox;
							BOOST_FOREACH(ptree::value_type &elem, pt.get_child("pbox")) {
								pbox.push_back(static_cast<int>(elem.second.get_value<float>()));
							}
							outfile << image_name << " " << label << " " << score;
							outfile << " " << pbox[0] << " " << pbox[1];
							outfile << " " << pbox[2] << " " << pbox[3];
							outfile << " " << pbox[4] << " " << pbox[5];
							outfile << " " << pbox[6] << " " << pbox[7];
							outfile << std::endl;
						}
					}
					name_count_ = 0;
					detections_.clear();
				}
			}
		}
		if (visualize_) {
#ifdef USE_OPENCV
			vector<cv::Mat> cv_imgs;
			this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
			vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
			//*********************************************************************
			VisualizePBox(cv_imgs, top[0], visualize_threshold_, colors,
				label_to_display_name_, save_file_);
#endif  // USE_OPENCV
		}
	}

#ifdef CPU_ONLY
	STUB_GPU_FORWARD(DetectionOutputPboxLayer, Forward);
#endif

	INSTANTIATE_CLASS(DetectionOutputPboxLayer);
	REGISTER_LAYER_CLASS(DetectionOutputPbox);

}  // namespace caffe
