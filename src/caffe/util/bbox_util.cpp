#include <algorithm>
#include <csignal>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cmath>


#include "boost/iterator/counting_iterator.hpp"

#include "caffe/util/bbox_util.hpp"

namespace caffe {

bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  return bbox1.score() > bbox2.score();
}

bool SortPBoxAscend(const NormalizedPBox& bbox1, const NormalizedPBox& bbox2) {
  return bbox1.score() < bbox2.score();
}

bool SortPBoxDescend(const NormalizedPBox& bbox1, const NormalizedPBox& bbox2) {
  return bbox1.score() > bbox2.score();
}
template <typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
                         const pair<float, T>& pair2) {
  return pair1.first < pair2.first;
}

// Explicit initialization.
template bool SortScorePairAscend(const pair<float, int>& pair1,
                                  const pair<float, int>& pair2);
template bool SortScorePairAscend(const pair<float, pair<int, int> >& pair1,
                                  const pair<float, pair<int, int> >& pair2);

template <typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
                          const pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
                                   const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
                                   const pair<float, pair<int, int> >& pair2);

NormalizedBBox UnitBBox() {
  NormalizedBBox unit_bbox;
  unit_bbox.set_xmin(0.);
  unit_bbox.set_ymin(0.);
  unit_bbox.set_xmax(1.);
  unit_bbox.set_ymax(1.);
  return unit_bbox;
}

bool IsCrossBoundaryBBox(const NormalizedBBox& bbox) {
  return bbox.xmin() < 0 || bbox.xmin() > 1 ||
      bbox.ymin() < 0 || bbox.ymin() > 1 ||
      bbox.xmax() < 0 || bbox.xmax() > 1 ||
      bbox.ymax() < 0 || bbox.ymax() > 1;
}

//********************************************************
bool IsCrossBoundaryPBox(const NormalizedPBox& pbox) {
	return std::max(std::max(std::max(pbox.ltopx(), pbox.rtopx()), pbox.lbottomx()), pbox.rbottomx()) < 0 ||
		std::max(std::max(std::max(pbox.ltopy(), pbox.rtopy()), pbox.lbottomy()), pbox.rbottomy()) < 0 ||
		std::min(std::min(std::min(pbox.ltopx(), pbox.rtopx()), pbox.lbottomx()), pbox.rbottomx()) < 0 ||
		std::min(std::min(std::min(pbox.ltopy(), pbox.rtopy()), pbox.lbottomy()), pbox.rbottomy()) < 0 ||
		std::max(std::max(std::max(pbox.ltopx(), pbox.rtopx()), pbox.lbottomx()), pbox.rbottomx()) > 1 ||
		std::max(std::max(std::max(pbox.ltopy(), pbox.rtopy()), pbox.lbottomy()), pbox.rbottomy()) > 1 ||
		std::min(std::min(std::min(pbox.ltopx(), pbox.rtopx()), pbox.lbottomx()), pbox.rbottomx()) > 1 ||
		std::min(std::min(std::min(pbox.ltopy(), pbox.rtopy()), pbox.lbottomy()), pbox.rbottomy()) > 1;
	/*return bbox.xmin() < 0 || bbox.xmin() > 1 ||
		bbox.ymin() < 0 || bbox.ymin() > 1 ||
		bbox.xmax() < 0 || bbox.xmax() > 1 ||
		bbox.ymax() < 0 || bbox.ymax() > 1;*/
}

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
                   NormalizedBBox* intersect_bbox) {
  if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() ||
      bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
    // Return [0, 0, 0, 0] if there is no intersection.
    intersect_bbox->set_xmin(0);
    intersect_bbox->set_ymin(0);
    intersect_bbox->set_xmax(0);
    intersect_bbox->set_ymax(0);
  } else {
    intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
    intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
    intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
    intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
  }
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
  if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0;
  } else {
    if (bbox.has_size()) {
      return bbox.size();
    } else {
      float width = bbox.xmax() - bbox.xmin();
      float height = bbox.ymax() - bbox.ymin();
      if (normalized) {
        return width * height;
      } else {
        // If bbox is not within range [0, 1].
        return (width + 1) * (height + 1);
      }
    }
  }
}


float PBoxSize(const NormalizedPBox& pbox, const bool normalized) {
	float ltx, lty, rbx, rby, lbx, lby, rtx,rty;
    ltx = pbox.ltopx(); lty = pbox.ltopy();
    rbx = pbox.rbottomx(); rby = pbox.rbottomy();
    lbx = pbox.lbottomx(); lby = pbox.lbottomy();
    rtx = pbox.rtopx();  rty = pbox.rtopy();
    float acad, bcbd, cacb, dadb;
    acad = ((rtx - ltx)*(lby-lty) - (lbx - ltx)*(rty - lty));
    bcbd = ((rtx - rbx)*(lby-rby)- (lbx-rbx)*(rty-rby));
    cacb = ((ltx - rtx)*(rby-rty)-(rbx-rtx)*(lty-rty));
    dadb = ((ltx-lbx)*(rby-lby)- (rbx-lbx)*(lty-lby));
	if (acad*bcbd>0 || cacb*dadb>0) {
	// If pbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	}
	else {
		if (pbox.has_size()) {
			return pbox.size();
		}
		else {
			float ltrbx = pbox.ltopx() - pbox.rbottomx();
			float ltrby = pbox.ltopy() - pbox.rbottomy();
			float lbrtx = pbox.rtopx() - pbox.lbottomx();
			float lbrty = pbox.rtopy() - pbox.lbottomy();
			//float width = bbox.xmax() - bbox.xmin();
			//float height = bbox.ymax() - bbox.ymin();
			if (normalized) {

				float ltrb = sqrt(pow(ltrbx, 2.) + pow(ltrby, 2.));
				float lbrt = sqrt(pow(lbrtx, 2.) + pow(lbrty, 2.));
				float alpha = fabs(sin(atan2(ltrbx, ltrby) - atan2(lbrtx, lbrty)));
				//LOG(INFO) << ltrb << lbrt << alpha;
				return ltrb * lbrt*alpha / 2. ;
			}
			else {
				// If pbox is not within range [0, 1].
				++ltrbx;
				++ltrby;
				++lbrtx;
				++lbrty;
				float ltrb = sqrt(pow(ltrbx, 2.) + pow(ltrby, 2.));
				float lbrt = sqrt(pow(lbrtx, 2.) + pow(lbrty, 2.));
				float alpha = fabs(sin(atan2(ltrbx, ltrby) - atan2(lbrtx, lbrty)));
				return ltrb * lbrt*alpha / 2. ;
			}
		}
    }
}


template <typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized) {
  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
    // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
    return Dtype(0.);
  } else {
    const Dtype width = bbox[2] - bbox[0];
    const Dtype height = bbox[3] - bbox[1];
    if (normalized) {
      return width * height;
    } else {
      // If bbox is not within range [0, 1].
      return (width + 1) * (height + 1);
    }
  }
}

template float BBoxSize(const float* bbox, const bool normalized);
template double BBoxSize(const double* bbox, const bool normalized);

template <typename Dtype>
Dtype PBoxSize(const Dtype* pbox, const bool normalized) {
	
  Dtype ltx, lty, rbx, rby, lbx, lby, rtx,rty;
    ltx = pbox[0]; lty = pbox[1];
    rbx = pbox[4]; rby = pbox[5];
    lbx = pbox[2]; lby = pbox[3];
    rtx = pbox[6];  rty = pbox[7];
    Dtype acad, bcbd, cacb, dadb;
    acad = ((rtx - ltx)*(lby-lty) - (lbx - ltx)*(rty - lty));
    bcbd = ((rtx - rbx)*(lby-rby)- (lbx-rbx)*(rty-rby));
    cacb = ((ltx - rtx)*(rby-rty)-(rbx-rtx)*(lty-rty));
    dadb = ((ltx-lbx)*(rby-lby)- (rbx-lbx)*(lty-lby));
	if (acad*bcbd>0 || cacb*dadb>0) {
	// If pbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	}
  else {
			const Dtype ltrbx = pbox[0] - pbox[4];
			const Dtype ltrby = pbox[1] - pbox[5];
			const Dtype lbrtx = pbox[6] - pbox[2];
			const Dtype lbrty = pbox[7] - pbox[3];
			//float width = bbox.xmax() - bbox.xmin();
			//float height = bbox.ymax() - bbox.ymin();
			if (normalized) {

				Dtype ltrb = sqrtf(powf(ltrbx, 2.f) + powf(ltrby, 2.f));
				Dtype lbrt = sqrtf(powf(lbrtx, 2.f) + powf(lbrty, 2.f));
				Dtype alpha = fabsf(sinf(atan2f(ltrbx, ltrby) - atan2f(lbrtx, lbrty)));
				return ltrb * lbrt*alpha / 2.f;
			}
			else {
				// If pbox is not within range [0, 1].
				Dtype ltrb = sqrtf(powf(ltrbx+1, 2.f) + powf(ltrby+1, 2.f));
				Dtype lbrt = sqrtf(powf(lbrtx+1, 2.f) + powf(lbrty+1, 2.f));
				Dtype alpha = fabsf(sinf(atan2f(ltrbx+1, ltrby+1) - atan2f(lbrtx+1, lbrty+1)));
				return ltrb * lbrt*alpha / 2.f;
    }
  }
}

template float PBoxSize(const float* pbox, const bool normalized); 
template double PBoxSize(const double* pbox, const bool normalized); 

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}
//pbox version
void ClipPBox(const NormalizedPBox& pbox, NormalizedPBox* clip_pbox) {
	clip_pbox->set_ltopx(std::max(std::min(pbox.ltopx(), 1.f), 0.f));
	clip_pbox->set_ltopy(std::max(std::min(pbox.ltopy(), 1.f), 0.f));
	clip_pbox->set_lbottomx(std::max(std::min(pbox.lbottomx(), 1.f), 0.f));
	clip_pbox->set_lbottomy(std::max(std::min(pbox.lbottomy(), 1.f), 0.f));
	clip_pbox->set_rbottomx(std::max(std::min(pbox.rbottomx(), 1.f), 0.f));
	clip_pbox->set_rbottomy(std::max(std::min(pbox.rbottomy(), 1.f), 0.f));
	clip_pbox->set_rtopx(std::max(std::min(pbox.rtopx(), 1.f), 0.f));
	clip_pbox->set_rtopy(std::max(std::min(pbox.rtopy(), 1.f), 0.f));
/*
	clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
	clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
	clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
	*/
	clip_pbox->clear_size();
	clip_pbox->set_size(PBoxSize(*clip_pbox));
	clip_pbox->set_difficult(pbox.difficult());
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
              NormalizedBBox* clip_bbox) {
  clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), width), 0.f));
  clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), height), 0.f));
  clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), width), 0.f));
  clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), height), 0.f));
  clip_bbox->clear_size();
  clip_bbox->set_size(BBoxSize(*clip_bbox));
  clip_bbox->set_difficult(bbox.difficult());
}

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
               NormalizedBBox* scale_bbox) {
  scale_bbox->set_xmin(bbox.xmin() * width);
  scale_bbox->set_ymin(bbox.ymin() * height);
  scale_bbox->set_xmax(bbox.xmax() * width);
  scale_bbox->set_ymax(bbox.ymax() * height);
  scale_bbox->clear_size();
  bool normalized = !(width > 1 || height > 1);
  scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
  scale_bbox->set_difficult(bbox.difficult());
}

void ScalePBox(const NormalizedPBox& pbox, const int height, const int width,
	NormalizedPBox* scale_pbox) {
	scale_pbox->set_ltopx(pbox.ltopx() * width);
	scale_pbox->set_ltopy(pbox.ltopy() * height);
	scale_pbox->set_lbottomx(pbox.lbottomx() * width);
	scale_pbox->set_lbottomy(pbox.lbottomy() * height);
	scale_pbox->set_rbottomx(pbox.rbottomx() * width);
	scale_pbox->set_rbottomy(pbox.rbottomy() * height);
	scale_pbox->set_rtopx(pbox.rtopx() * width);
	scale_pbox->set_rtopy(pbox.rtopy() * height);
/*
	scale_bbox->set_xmin(bbox.xmin() * width);
	scale_bbox->set_ymin(bbox.ymin() * height);
	scale_bbox->set_xmax(bbox.xmax() * width);
	scale_bbox->set_ymax(bbox.ymax() * height);*/
	scale_pbox->clear_size();
	bool normalized = !(width > 1 || height > 1);
	scale_pbox->set_size(PBoxSize(*scale_pbox, normalized));
	scale_pbox->set_difficult(pbox.difficult());
}

void OutputBBox(const NormalizedBBox& bbox, const pair<int, int>& img_size,
                const bool has_resize, const ResizeParameter& resize_param,
                NormalizedBBox* out_bbox) {
  const int height = img_size.first;
  const int width = img_size.second;
  NormalizedBBox temp_bbox = bbox;
  if (has_resize && resize_param.resize_mode()) {
    float resize_height = resize_param.height();
    CHECK_GT(resize_height, 0);
    float resize_width = resize_param.width();
    CHECK_GT(resize_width, 0);
    float resize_aspect = resize_width / resize_height;
    int height_scale = resize_param.height_scale();
    int width_scale = resize_param.width_scale();
    float aspect = static_cast<float>(width) / height;

    float padding;
    NormalizedBBox source_bbox;
    switch (resize_param.resize_mode()) {
      case ResizeParameter_Resize_mode_WARP:
        ClipBBox(temp_bbox, &temp_bbox);
        ScaleBBox(temp_bbox, height, width, out_bbox);
        break;
      case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
        if (aspect > resize_aspect) {
          padding = (resize_height - resize_width / aspect) / 2;
          source_bbox.set_xmin(0.);
          source_bbox.set_ymin(padding / resize_height);
          source_bbox.set_xmax(1.);
          source_bbox.set_ymax(1. - padding / resize_height);
        } else {
          padding = (resize_width - resize_height * aspect) / 2;
          source_bbox.set_xmin(padding / resize_width);
          source_bbox.set_ymin(0.);
          source_bbox.set_xmax(1. - padding / resize_width);
          source_bbox.set_ymax(1.);
        }
        ProjectBBox(source_bbox, bbox, &temp_bbox);
        ClipBBox(temp_bbox, &temp_bbox);
        ScaleBBox(temp_bbox, height, width, out_bbox);
        break;
      case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
        if (height_scale == 0 || width_scale == 0) {
          ClipBBox(temp_bbox, &temp_bbox);
          ScaleBBox(temp_bbox, height, width, out_bbox);
        } else {
          ScaleBBox(temp_bbox, height_scale, width_scale, out_bbox);
          ClipBBox(*out_bbox, height, width, out_bbox);
        }
        break;
      default:
        LOG(FATAL) << "Unknown resize mode.";
    }
  } else {
    // Clip the normalized bbox first.
    ClipBBox(temp_bbox, &temp_bbox);
    // Scale the bbox according to the original image size.
    ScaleBBox(temp_bbox, height, width, out_bbox);
  }
}

void OutputPBox(const NormalizedPBox& pbox, const pair<int, int>& img_size,
	const bool has_resize, const ResizeParameter& resize_param,
	NormalizedPBox* out_pbox) {
	const int height = img_size.first;
	const int width = img_size.second;
	NormalizedPBox temp_pbox = pbox;
	if (has_resize && resize_param.resize_mode()) {
		/*float resize_height = resize_param.height();
		CHECK_GT(resize_height, 0);
		float resize_width = resize_param.width();
		CHECK_GT(resize_width, 0);
		float resize_aspect = resize_width / resize_height;
		int height_scale = resize_param.height_scale();
		int width_scale = resize_param.width_scale();
		float aspect = static_cast<float>(width) / height;

		float padding;
		NormalizedPBox source_pbox;
		switch (resize_param.resize_mode()) {
		case ResizeParameter_Resize_mode_WARP:
			ClipBBox(temp_bbox, &temp_bbox);
			ScaleBBox(temp_bbox, height, width, out_bbox);
			break;
		case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
			if (aspect > resize_aspect) {
				padding = (resize_height - resize_width / aspect) / 2;
				source_bbox.set_xmin(0.);
				source_bbox.set_ymin(padding / resize_height);
				source_bbox.set_xmax(1.);
				source_bbox.set_ymax(1. - padding / resize_height);
			}
			else {
				padding = (resize_width - resize_height * aspect) / 2;
				source_bbox.set_xmin(padding / resize_width);
				source_bbox.set_ymin(0.);
				source_bbox.set_xmax(1. - padding / resize_width);
				source_bbox.set_ymax(1.);
			}
			ProjectBBox(source_bbox, bbox, &temp_bbox);
			ClipBBox(temp_bbox, &temp_bbox);
			ScaleBBox(temp_bbox, height, width, out_bbox);
			break;
		case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
			if (height_scale == 0 || width_scale == 0) {
				ClipBBox(temp_bbox, &temp_bbox);
				ScaleBBox(temp_bbox, height, width, out_bbox);
			}
			else {
				ScaleBBox(temp_bbox, height_scale, width_scale, out_bbox);
				ClipBBox(*out_bbox, height, width, out_bbox);
			}
			break;
		default:
			LOG(FATAL) << "Unknown resize mode.";
		}*/
	}
	else {
		// Clip the normalized bbox first.
		ClipPBox(temp_pbox, &temp_pbox);
		// Scale the bbox according to the original image size.
		ScalePBox(temp_pbox, height, width, out_pbox);
	}
}

void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                NormalizedBBox* loc_bbox) {
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
  loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
  loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
  loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
  loc_bbox->set_difficult(bbox.difficult());
}

bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
                 NormalizedBBox* proj_bbox) {
  if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin() ||
      bbox.ymin() >= src_bbox.ymax() || bbox.ymax() <= src_bbox.ymin()) {
    return false;
  }
  float src_width = src_bbox.xmax() - src_bbox.xmin();
  float src_height = src_bbox.ymax() - src_bbox.ymin();
  proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
  proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
  proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
  proj_bbox->set_difficult(bbox.difficult());
  ClipBBox(*proj_bbox, proj_bbox);
  if (BBoxSize(*proj_bbox) > 0) {
    return true;
  } else {
    return false;
  }
}

bool ProjectPBox(const NormalizedBBox& src_bbox, const NormalizedPBox& pbox,
	NormalizedPBox* proj_pbox) {
	//*****************************************************************************
	//only consider that the src_bbox contains the pbox
	if (pbox.ltopx() >= src_bbox.xmax() || pbox.rtopx() >= src_bbox.xmax() ||
		pbox.lbottomx() >= src_bbox.xmax() || pbox.rbottomx() >= src_bbox.xmax() ||
		pbox.ltopx() <= src_bbox.xmin() || pbox.rtopx() <= src_bbox.xmin() ||
		pbox.lbottomx() <= src_bbox.xmin() || pbox.rbottomx() <= src_bbox.xmin() ||
		pbox.ltopy() >= src_bbox.ymax() || pbox.rtopy() >= src_bbox.ymax() ||
		pbox.lbottomy() >= src_bbox.ymax() || pbox.rbottomy() >= src_bbox.ymax() ||
		pbox.ltopy() <= src_bbox.ymin() || pbox.rtopy() <= src_bbox.ymin() ||
		pbox.lbottomy() <= src_bbox.ymin() || pbox.rbottomy() <= src_bbox.ymin()) {
		return false;
	}
	float src_width = src_bbox.xmax() - src_bbox.xmin();
	float src_height = src_bbox.ymax() - src_bbox.ymin();
	proj_pbox->set_ltopx((pbox.ltopx() - src_bbox.xmin()) / src_width);
	proj_pbox->set_lbottomx((pbox.lbottomx() - src_bbox.xmin()) / src_width);
	proj_pbox->set_rtopx((pbox.rtopx() - src_bbox.xmin()) / src_width);
	proj_pbox->set_rbottomx((pbox.rbottomx() - src_bbox.xmin()) / src_width);
	proj_pbox->set_ltopy((pbox.ltopy() - src_bbox.ymin()) / src_height);
	proj_pbox->set_lbottomy((pbox.lbottomy() - src_bbox.ymin()) / src_height);
	proj_pbox->set_rtopy((pbox.rtopy() - src_bbox.ymin()) / src_height);
	proj_pbox->set_rbottomy((pbox.rbottomy() - src_bbox.ymin()) / src_height);
	//LOG(FATAL) << proj_pbox->lbottomy() << " " << proj_pbox->rbottomy();

	//proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
	//proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
	//proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
	//proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
	proj_pbox->set_difficult(pbox.difficult());
	ClipPBox(*proj_pbox, proj_pbox);
	//LOG(FATAL) << PBoxSize(*proj_pbox);
	if (PBoxSize(*proj_pbox) > 0) {
		return true;
	}
	else {
		return false;
	}
}

void ExtrapolateBBox(const ResizeParameter& param, const int height,
    const int width, const NormalizedBBox& crop_bbox, NormalizedBBox* bbox) {
  float height_scale = param.height_scale();
  float width_scale = param.width_scale();
  if (height_scale > 0 && width_scale > 0 &&
      param.resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
    float orig_aspect = static_cast<float>(width) / height;
    float resize_height = param.height();
    float resize_width = param.width();
    float resize_aspect = resize_width / resize_height;
    if (orig_aspect < resize_aspect) {
      resize_height = resize_width / orig_aspect;
    } else {
      resize_width = resize_height * orig_aspect;
    }
    float crop_height = resize_height * (crop_bbox.ymax() - crop_bbox.ymin());
    float crop_width = resize_width * (crop_bbox.xmax() - crop_bbox.xmin());
    CHECK_GE(crop_width, width_scale);
    CHECK_GE(crop_height, height_scale);
    bbox->set_xmin(bbox->xmin() * crop_width / width_scale);
    bbox->set_xmax(bbox->xmax() * crop_width / width_scale);
    bbox->set_ymin(bbox->ymin() * crop_height / height_scale);
    bbox->set_ymax(bbox->ymax() * crop_height / height_scale);
  }
}
//**************************************************************************
//only support the resize mode : fit small size
void ExtrapolatePBox(const ResizeParameter& param, const int height,
	const int width, const NormalizedBBox& crop_bbox, NormalizedPBox* pbox) {
	float height_scale = param.height_scale();
	float width_scale = param.width_scale();
	if (height_scale > 0 && width_scale > 0 &&
		param.resize_mode() == ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
		float orig_aspect = static_cast<float>(width) / height;
		float resize_height = param.height();
		float resize_width = param.width();
		float resize_aspect = resize_width / resize_height;
		if (orig_aspect < resize_aspect) {
			resize_height = resize_width / orig_aspect;
		}
		else {
			resize_width = resize_height * orig_aspect;
		}
		float crop_height = resize_height * (crop_bbox.ymax() - crop_bbox.ymin());
		float crop_width = resize_width * (crop_bbox.xmax() - crop_bbox.xmin());
		CHECK_GE(crop_width, width_scale);
		CHECK_GE(crop_height, height_scale);
		pbox->set_ltopx(pbox->ltopx() * crop_width / width_scale);
		pbox->set_ltopy(pbox->ltopy() * crop_width / width_scale);
		pbox->set_lbottomx(pbox->lbottomx() * crop_width / width_scale);
		pbox->set_lbottomy(pbox->lbottomy() * crop_width / width_scale);
		pbox->set_rbottomx(pbox->rbottomx() * crop_width / width_scale);
		pbox->set_rbottomy(pbox->rbottomy() * crop_width / width_scale);
		pbox->set_rtopx(pbox->rtopx() * crop_width / width_scale);
		pbox->set_rtopy(pbox->rtopy() * crop_width / width_scale);

	}
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
	const bool normalized) {
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_width, intersect_height;
	if (normalized) {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
	}
	else {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
	}
	if (intersect_width > 0 && intersect_height > 0) {
		float intersect_size = intersect_width * intersect_height;
		float bbox1_size = BBoxSize(bbox1);
		float bbox2_size = BBoxSize(bbox2);
		return intersect_size / (bbox1_size + bbox2_size - intersect_size);
	}
	else {
		return 0.;
	}
}
//********************************************************************************
float JaccardOverlapPB(const NormalizedBBox& sampled, const NormalizedPBox& object,
	const bool normalized) {
	//NormalizedBBox intersect_bbox;
	//IntersectBBox(sampled, object, &intersect_bbox);
	/*if (std::min(object.ltopx(), object.lbottomx()) > sampled.xmax() ||
		std::min(object.ltopy(), object.rtopy()) > sampled.ymax() ||
		std::max(object.rtopx(), object.rbottomx()) < sampled.xmin() ||
		std::max(object.lbottomy(), object.rbottomy) < sampled.ymin()) {
		return 0;
	}*/
	float xmin, xmax, ymin, ymax;
	xmin = std::min(std::min(std::min(object.ltopx(), object.lbottomx()), object.rbottomx()), object.rtopx());
	xmax = std::max(std::max(std::max(object.ltopx(), object.lbottomx()), object.rbottomx()), object.rtopx());
	ymin = std::min(std::min(std::min(object.ltopy(), object.lbottomy()), object.rbottomy()), object.rtopy());
	ymax = std::max(std::max(std::max(object.ltopy(), object.lbottomy()), object.rbottomy()), object.rtopy());
	if (xmin > sampled.xmin() &&
		ymin > sampled.ymin() &&
		xmax < sampled.xmax() &&
		ymax < sampled.ymax())
	{

		float objarea = PBoxSize(object);
		float bbox1_size = BBoxSize(sampled);
		return objarea / bbox1_size;
	}
	else {
		return 0. ;
	}
	//float intersect_width, intersect_height;
	//if (normalized) {
	//	intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
	//	intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
	//}
	//else {
	//	intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
	//	intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
	//}
	//if (intersect_width > 0 && intersect_height > 0) {
	//	float intersect_size = intersect_width * intersect_height;
	//	float bbox1_size = BBoxSize(sampled);
	//	//****************************************
	//	float bbox2_size = BBoxSize(object);
	//	return intersect_size / (bbox1_size + bbox2_size - intersect_size);
	//}
	//else {
	//	return 0.;
	//}
}
//*****************************************************************************
float JaccardOverlapPbox(const NormalizedPBox& pbox1, const NormalizedPBox& pbox2,
	const bool normalized) {
	//*********************************************
	/*multimap<float,float> x1y1 = { {pbox1.ltopx(),pbox1.ltopy()},{pbox1.lbottomx(),pbox1.lbottomy()},
									{pbox1.rbottomx(),pbox1.rbottomy()},{pbox1.rtopx(),pbox1.rtopy()} };
	vector<vector<float>> v1 = { { pbox1.ltopx(),pbox1.ltopy() },{ pbox1.lbottomx(),pbox1.lbottomy() },
								{ pbox1.rbottomx(),pbox1.rbottomy() },{ pbox1.rtopx(),pbox1.rtopy() } };
	multimap<float, float> x2y2 = { { pbox2.ltopx(),pbox2.ltopy() },{ pbox2.lbottomx(),pbox2.lbottomy() },
									{ pbox2.rbottomx(),pbox2.rbottomy() },{ pbox2.rtopx(),pbox2.rtopy() } };
	vector<vector<float>> v2 = { { pbox2.ltopx(),pbox2.ltopy() },{ pbox2.lbottomx(),pbox2.lbottomy() },
									{ pbox2.rbottomx(),pbox2.rbottomy() },{ pbox2.rtopx(),pbox2.rtopy() } };
	float leftmostx1 = std::min(std::min(std::min(pbox1.ltopx(), pbox1.rtopx()), pbox1.lbottomx()), pbox1.rbottomx());
	float leftmosty1 = x1y1.find(leftmostx1)->second;
	float leftmostx2 = std::min(std::min(std::min(pbox2.ltopx(), pbox2.rtopx()), pbox2.lbottomx()), pbox2.rbottomx());
	float leftmosty2 = x2y2.find(leftmostx2)->second;
	multimap<float, float>::iterator t1, t2 , t;
	t1 = x1y1.find(leftmostx1);
	t2 = x2y2.find(leftmostx2);
	float k1, k2;
	vector<vector<float>>::iterator vt1, vt2;
	while (t1 < x1y1.end() && t2 < x2y2.end())
	{
		for (t = t1+1; t < x1y1.end(); ++t)
		{
			float temp = 0.f;
			vector<float> p1 = { leftmostx1, leftmosty1 };
			vector<float> p2 = { leftmostx2, leftmosty2 };

			vt1 = v1.find(p1);
			if ( vt1 == v1.begin())
			{
				temp = (v1.end()->second - leftmosty1) / (v1.end()->first - leftmostx1);
				k1 = ((v1.begin()++)->second - leftmosty1) / ((v1.begin()++)->first - leftmostx1);
			}
			else if (vt1 == (v1.end()--))
			{
				temp = (v1.begin()->second - leftmosty1) / (v1.begin()->first - leftmostx1);
				k1 = ((v1.end()-2)->second - leftmosty1) / ((v1.end()-2)->first - leftmostx1);
			}
			else
			{
				temp = ((vt1 - 1)->second - leftmosty1) / ((vt1 - 1)->first - leftmostx1);
				k1 = ((vt1 + 1)->second - leftmosty1) / (vt1 + 1)->first - leftmostx1);
			}
			k1 = std::max(temp, k1);

			vt2 = v2.find(p2);
			if (vt2 == v2.begin())
			{
				temp = (v2.end()->second - leftmosty2) / (v2.end()->first - leftmostx2);
				k2 = ((v2.begin()+1)->second - leftmosty2) / ((v2.begin()+1)->first - leftmostx2);
			}
			else if (vt2 == (v2.end()--))
			{
				temp = (v2.begin()->second - leftmosty2) / (v2.begin()->first - leftmostx2);
				k2 = ((v2.end() - 2)->second - leftmosty2) / ((v2.end() - 2)->first - leftmostx2);
			}
			else
			{
				temp = ((vt2 - 1)->second - leftmosty2) / ((vt2 - 1))->first - leftmostx2);
				k2 = ((vt2 + 1)->second - leftmosty2) / ((vt2 + 1)->first - leftmostx2);
			}
			k2 = std::max(temp, k2);

			if (k1 > k2)

			if (t->first == t1->first)
				break;

			if(k1)
			k1 = (t->second - t1->second) / (t->first - t1->first);
			temp = k1;


		}
		for (t = t2; t < x2y2.end(); ++t)
		{
			if (t->first == t2->first)
				break;
			else if (t->second > t2->second)
				break;
			else if (t==\A3\A8x2y2.end()-1\A3\A9)

			else
				k1 = (t->second - t2->second) / (t->first - t2->first);

		}
	}*/
	float d1, d2, d3, d4;
	d1 = sqrt((pow((pbox1.ltopx() - pbox2.ltopx()), 2.f) + pow((pbox1.ltopy() - pbox2.ltopy()), 2.f)));
	d2 = sqrt((pow((pbox1.lbottomx() - pbox2.lbottomx()), 2.f) + pow((pbox1.lbottomy() - pbox2.lbottomy()), 2.f)));
	d3 = sqrt((pow((pbox1.rbottomx() - pbox2.rbottomx()), 2.f) + pow((pbox1.rbottomy() - pbox2.rbottomy()), 2.f)));
	d4 = sqrt((pow((pbox1.rtopx() - pbox2.rtopx()), 2.f) + pow((pbox1.rtopy() - pbox2.rtopy()), 2.f)));

	float d = std::max(std::max(std::max(d1, d2), d3), d4);
	//LOG(INFO) << "d: " << d;

	d1 = sqrt((pow((pbox1.ltopx() - pbox1.lbottomx()), 2.f) + pow((pbox1.ltopy() - pbox1.lbottomy()), 2.f)));
	d2 = sqrt((pow((pbox1.lbottomx() - pbox1.rbottomx()), 2.f) + pow((pbox1.lbottomy() - pbox1.rbottomy()), 2.f)));
	d3 = sqrt((pow((pbox1.rbottomx() - pbox1.rtopx()), 2.f) + pow((pbox1.rbottomy() - pbox1.rtopy()), 2.f)));
	d4 = sqrt((pow((pbox1.rtopx() - pbox1.ltopx()), 2.f) + pow((pbox1.rtopy() - pbox1.ltopy()), 2.f)));

	float D = std::min(std::min(std::min(d1, d2), d3), d4);
	//LOG(INFO) << d1 << " " << d2 << " " << d3 << " " << d4;
	//LOG(INFO) << "D: " << D; 

	return 1.f - expf(-D / d);
	/*float intersect_width, intersect_height;
	if (normalized) {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
	}
	else {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
	}
	if (intersect_width > 0 && intersect_height > 0) {
		float intersect_size = intersect_width * intersect_height;
		float pbox1_size = PBoxSize(pbox1);
		float pbox2_size = PBoxSize(pbox2);
		return intersect_size / (pbox1_size + pbox2_size - intersect_size);
	}
	else {
		return 0.;
	}*/
}


template <typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSize(bbox1);
    const Dtype bbox2_size = BBoxSize(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template float JaccardOverlap(const float* bbox1, const float* bbox2);
template double JaccardOverlap(const double* bbox1, const double* bbox2);

template <typename Dtype>
Dtype JaccardOverlapPbox(const Dtype* pbox1, const Dtype* pbox2) {
	const Dtype d1 = sqrt((pow((pbox1[0] - pbox2[0]), 2.f) + pow((pbox1[1] - pbox2[1]), 2.f)));
	const Dtype d2 = sqrt((pow((pbox1[2] - pbox2[2]), 2.f) + pow((pbox1[3] - pbox2[3]), 2.f)));
	const Dtype d3 = sqrt((pow((pbox1[4] - pbox2[4]), 2.f) + pow((pbox1[5] - pbox2[5]), 2.f)));
	const Dtype d4 = sqrt((pow((pbox1[6] - pbox2[6]), 2.f) + pow((pbox1[7] - pbox2[7]), 2.f)));

	const Dtype d = std::max(std::max(std::max(d1, d2), d3), d4);

	const Dtype D1 = sqrt((pow((pbox1[0] - pbox1[2]), 2.f) + pow((pbox1[1] - pbox1[3]), 2.f)));
	const Dtype D2 = sqrt((pow((pbox1[2] - pbox1[4]), 2.f) + pow((pbox1[3] - pbox1[5]), 2.f)));
	const Dtype D3 = sqrt((pow((pbox1[4] - pbox1[6]), 2.f) + pow((pbox1[5] - pbox1[7]), 2.f)));
	const Dtype D4 = sqrt((pow((pbox1[6] - pbox1[0]), 2.f) + pow((pbox1[7] - pbox1[1]), 2.f)));

	const Dtype D = std::min(std::min(std::min(D1, D2), D3), D4);

	return 1.f - expf(-D / d);
}

template float JaccardOverlapPbox(const float* pbox1, const float* pbox2);
template double JaccardOverlapPbox(const double* pbox1, const double* pbox2);


float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
  NormalizedBBox intersect_bbox;
  IntersectBBox(bbox1, bbox2, &intersect_bbox);
  float intersect_size = BBoxSize(intersect_bbox);
  if (intersect_size > 0) {
    float bbox1_size = BBoxSize(bbox1);
    return intersect_size / bbox1_size;
  } else {
    return 0.;
  }
}

bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
                        const NormalizedBBox& bbox,
                        const EmitConstraint& emit_constraint) {
  EmitType emit_type = emit_constraint.emit_type();
  if (emit_type == EmitConstraint_EmitType_CENTER) {
    float x_center = (bbox.xmin() + bbox.xmax()) / 2;
    float y_center = (bbox.ymin() + bbox.ymax()) / 2;
    if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
        y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
      return true;
    } else {
      return false;
    }
  } else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
    float bbox_coverage = BBoxCoverage(bbox, src_bbox);
    return bbox_coverage > emit_constraint.emit_overlap();
  } else {
    LOG(FATAL) << "Unknown emit type.";
    return false;
  }
}

bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
                        const NormalizedPBox& pbox,
                        const EmitConstraint& emit_constraint) {
  /*EmitType emit_type = emit_constraint.emit_type();*/
  //if (emit_type == EmitConstraint_EmitType_CENTER) {
    //float x_center = (pbox.xmin() + pbox.xmax()) / 2;
    //float y_center = (pbox.ymin() + pbox.ymax()) / 2;
    //if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax() &&
        //y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
      //return true;
    //} else {
      //return false;
    //}
  //} else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
    //float bbox_coverage = BBoxCoverage(bbox, src_bbox);
    //return bbox_coverage > emit_constraint.emit_overlap();
  //} else {
    //LOG(FATAL) << "Unknown emit type.";
    //return false;
  /*}*/
    return true;
}


void EncodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool encode_variance_in_target,
    const NormalizedBBox& bbox, NormalizedBBox* encode_bbox) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (encode_variance_in_target) {
      encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
      encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
      encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
      encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (int i = 0; i < prior_variance.size(); ++i) {
        CHECK_GT(prior_variance[i], 0);
      }
      encode_bbox->set_xmin(
          (bbox.xmin() - prior_bbox.xmin()) / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox.ymin() - prior_bbox.ymin()) / prior_variance[1]);
      encode_bbox->set_xmax(
          (bbox.xmax() - prior_bbox.xmax()) / prior_variance[2]);
      encode_bbox->set_ymax(
          (bbox.ymax() - prior_bbox.ymax()) / prior_variance[3]);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    float bbox_width = bbox.xmax() - bbox.xmin();
    CHECK_GT(bbox_width, 0);
    float bbox_height = bbox.ymax() - bbox.ymin();
    CHECK_GT(bbox_height, 0);
    float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
    float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
      encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
      encode_bbox->set_xmax(log(bbox_width / prior_width));
      encode_bbox->set_ymax(log(bbox_height / prior_height));
    } else {
      // Encode variance in bbox.
      encode_bbox->set_xmin(
          (bbox_center_x - prior_center_x) / prior_width / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox_center_y - prior_center_y) / prior_height / prior_variance[1]);
      encode_bbox->set_xmax(
          log(bbox_width / prior_width) / prior_variance[2]);
      encode_bbox->set_ymax(
          log(bbox_height / prior_height) / prior_variance[3]);
    }
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    if (encode_variance_in_target) {
      encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_width);
      encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_height);
      encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_width);
      encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_height);
    } else {
      // Encode variance in bbox.
      CHECK_EQ(prior_variance.size(), 4);
      for (int i = 0; i < prior_variance.size(); ++i) {
        CHECK_GT(prior_variance[i], 0);
      }
      encode_bbox->set_xmin(
          (bbox.xmin() - prior_bbox.xmin()) / prior_width / prior_variance[0]);
      encode_bbox->set_ymin(
          (bbox.ymin() - prior_bbox.ymin()) / prior_height / prior_variance[1]);
      encode_bbox->set_xmax(
          (bbox.xmax() - prior_bbox.xmax()) / prior_width / prior_variance[2]);
      encode_bbox->set_ymax(
          (bbox.ymax() - prior_bbox.ymax()) / prior_height / prior_variance[3]);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
}

//*******************************************************************************
void EncodePBox(const NormalizedPBox& prior_pbox,
	const vector<float>& prior_variance, const PCodeType code_type,
	const bool encode_variance_in_target, const NormalizedPBox& pbox,
	NormalizedPBox* encode_pbox) {
	if (code_type == PriorPBoxParameter_CodeType_CORNER) {
		if (encode_variance_in_target) {
			encode_pbox->set_ltopx(pbox.ltopx() - prior_pbox.ltopx());
			encode_pbox->set_ltopy(pbox.ltopy() - prior_pbox.ltopy());
			encode_pbox->set_lbottomx(pbox.lbottomx() - prior_pbox.lbottomx());
			encode_pbox->set_lbottomy(pbox.lbottomy() - prior_pbox.lbottomy());
			encode_pbox->set_rbottomx(pbox.rbottomx() - prior_pbox.rbottomx());
			encode_pbox->set_rbottomy(pbox.rbottomy() - prior_pbox.rbottomy());
			encode_pbox->set_rtopx(pbox.rtopx() - prior_pbox.rtopx());
			encode_pbox->set_rtopy(pbox.rtopy() - prior_pbox.rtopy());
		/*	encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
			encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
			encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
			encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());*/
		}
		else {
			// Encode variance in pbox.
			CHECK_EQ(prior_variance.size(), 8);
			for (int i = 0; i < prior_variance.size(); ++i) {
				CHECK_GT(prior_variance[i], 0);
			}
			encode_pbox->set_ltopx(
				(pbox.ltopx() - prior_pbox.ltopx()) / prior_variance[0]);
			encode_pbox->set_ltopy(
				(pbox.ltopy() - prior_pbox.ltopy()) / prior_variance[1]);
			encode_pbox->set_lbottomx(
				(pbox.lbottomx() - prior_pbox.lbottomx()) / prior_variance[2]);
			encode_pbox->set_lbottomy(
				(pbox.lbottomy() - prior_pbox.lbottomy()) / prior_variance[3]);
			encode_pbox->set_rbottomx(
				(pbox.rbottomx() - prior_pbox.rbottomx()) / prior_variance[4]);
			encode_pbox->set_rbottomy(
				(pbox.rbottomy() - prior_pbox.rbottomy()) / prior_variance[5]);
			encode_pbox->set_rtopx(
				(pbox.rtopx() - prior_pbox.rtopx()) / prior_variance[6]);
			encode_pbox->set_rtopy(
				(pbox.rtopy() - prior_pbox.rtopy()) / prior_variance[7]);
		/*	encode_bbox->set_xmin(
				(bbox.xmin() - prior_bbox.xmin()) / prior_variance[0]);
			encode_bbox->set_ymin(
				(bbox.ymin() - prior_bbox.ymin()) / prior_variance[1]);
			encode_bbox->set_xmax(
				(bbox.xmax() - prior_bbox.xmax()) / prior_variance[2]);
			encode_bbox->set_ymax(
				(bbox.ymax() - prior_bbox.ymax()) / prior_variance[3]);*/
		}
	}
	//else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
	//	float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
	//	CHECK_GT(prior_width, 0);
	//	float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
	//	CHECK_GT(prior_height, 0);
	//	float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
	//	float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

	//	float bbox_width = bbox.xmax() - bbox.xmin();
	//	CHECK_GT(bbox_width, 0);
	//	float bbox_height = bbox.ymax() - bbox.ymin();
	//	CHECK_GT(bbox_height, 0);
	//	float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
	//	float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

	//	if (encode_variance_in_target) {
	//		encode_bbox->set_xmin((bbox_center_x - prior_center_x) / prior_width);
	//		encode_bbox->set_ymin((bbox_center_y - prior_center_y) / prior_height);
	//		encode_bbox->set_xmax(log(bbox_width / prior_width));
	//		encode_bbox->set_ymax(log(bbox_height / prior_height));
	//	}
	//	else {
	//		// Encode variance in bbox.
	//		encode_bbox->set_xmin(
	//			(bbox_center_x - prior_center_x) / prior_width / prior_variance[0]);
	//		encode_bbox->set_ymin(
	//			(bbox_center_y - prior_center_y) / prior_height / prior_variance[1]);
	//		encode_bbox->set_xmax(
	//			log(bbox_width / prior_width) / prior_variance[2]);
	//		encode_bbox->set_ymax(
	//			log(bbox_height / prior_height) / prior_variance[3]);
	//	}
	//}
	//else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
	//	float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
	//	CHECK_GT(prior_width, 0);
	//	float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
	//	CHECK_GT(prior_height, 0);
	//	if (encode_variance_in_target) {
	//		encode_bbox->set_xmin((bbox.xmin() - prior_bbox.xmin()) / prior_width);
	//		encode_bbox->set_ymin((bbox.ymin() - prior_bbox.ymin()) / prior_height);
	//		encode_bbox->set_xmax((bbox.xmax() - prior_bbox.xmax()) / prior_width);
	//		encode_bbox->set_ymax((bbox.ymax() - prior_bbox.ymax()) / prior_height);
	//	}
	//	else {
	//		// Encode variance in bbox.
	//		CHECK_EQ(prior_variance.size(), 4);
	//		for (int i = 0; i < prior_variance.size(); ++i) {
	//			CHECK_GT(prior_variance[i], 0);
	//		}
	//		encode_bbox->set_xmin(
	//			(bbox.xmin() - prior_bbox.xmin()) / prior_width / prior_variance[0]);
	//		encode_bbox->set_ymin(
	//			(bbox.ymin() - prior_bbox.ymin()) / prior_height / prior_variance[1]);
	//		encode_bbox->set_xmax(
	//			(bbox.xmax() - prior_bbox.xmax()) / prior_width / prior_variance[2]);
	//		encode_bbox->set_ymax(
	//			(bbox.ymax() - prior_bbox.ymax()) / prior_height / prior_variance[3]);
	//	}
	//}
	else {
		LOG(FATAL) << "Unknown LocLossType.";
	}
}

void DecodeBBox(
    const NormalizedBBox& prior_bbox, const vector<float>& prior_variance,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox) {
  if (code_type == PriorBoxParameter_CodeType_CORNER) {
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(
          prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
      decode_bbox->set_ymin(
          prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
      decode_bbox->set_xmax(
          prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
      decode_bbox->set_ymax(
          prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
    }
  } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
    float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

    float decode_bbox_center_x, decode_bbox_center_y;
    float decode_bbox_width, decode_bbox_height;
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to retore the offset
      // predictions.
      decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width = exp(bbox.xmax()) * prior_width;
      decode_bbox_height = exp(bbox.ymax()) * prior_height;
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox_center_x =
          prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
      decode_bbox_center_y =
          prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
      decode_bbox_width =
          exp(prior_variance[2] * bbox.xmax()) * prior_width;
      decode_bbox_height =
          exp(prior_variance[3] * bbox.ymax()) * prior_height;
    }

    decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
    decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
    decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
    decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
  } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
    float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
    CHECK_GT(prior_width, 0);
    float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
    CHECK_GT(prior_height, 0);
    if (variance_encoded_in_target) {
      // variance is encoded in target, we simply need to add the offset
      // predictions.
      decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin() * prior_width);
      decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin() * prior_height);
      decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax() * prior_width);
      decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax() * prior_height);
    } else {
      // variance is encoded in bbox, we need to scale the offset accordingly.
      decode_bbox->set_xmin(
          prior_bbox.xmin() + prior_variance[0] * bbox.xmin() * prior_width);
      decode_bbox->set_ymin(
          prior_bbox.ymin() + prior_variance[1] * bbox.ymin() * prior_height);
      decode_bbox->set_xmax(
          prior_bbox.xmax() + prior_variance[2] * bbox.xmax() * prior_width);
      decode_bbox->set_ymax(
          prior_bbox.ymax() + prior_variance[3] * bbox.ymax() * prior_height);
    }
  } else {
    LOG(FATAL) << "Unknown LocLossType.";
  }
  float bbox_size = BBoxSize(*decode_bbox);
  decode_bbox->set_size(bbox_size);
  if (clip_bbox) {
    ClipBBox(*decode_bbox, decode_bbox);
  }
}


void DecodeBBoxes(
	const vector<NormalizedBBox>& prior_bboxes,
	const vector<vector<float> >& prior_variances,
	const CodeType code_type, const bool variance_encoded_in_target,
	const bool clip_bbox, const vector<NormalizedBBox>& bboxes,
	vector<NormalizedBBox>* decode_bboxes) {
	CHECK_EQ(prior_bboxes.size(), prior_variances.size());
	CHECK_EQ(prior_bboxes.size(), bboxes.size());
	int num_bboxes = prior_bboxes.size();
	if (num_bboxes >= 1) {
		CHECK_EQ(prior_variances[0].size(), 4);
	}
	decode_bboxes->clear();
	for (int i = 0; i < num_bboxes; ++i) {
		NormalizedBBox decode_bbox;
		DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
			variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
		decode_bboxes->push_back(decode_bbox);
	}
}

//********************************************************************
void DecodePBox(
	const NormalizedPBox& prior_pbox, const vector<float>& prior_variance,
	const PCodeType code_type, const bool variance_encoded_in_target,
	const bool clip_pbox, const NormalizedPBox& pbox,
	NormalizedPBox* decode_pbox) {
	if (code_type == PriorPBoxParameter_CodeType_CORNER) {
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to add the offset
			// predictions.
			decode_pbox->set_ltopx(prior_pbox.ltopx() + pbox.ltopx());
			decode_pbox->set_ltopy(prior_pbox.ltopy() + pbox.ltopy());
			decode_pbox->set_lbottomx(prior_pbox.lbottomx() + pbox.lbottomx());
			decode_pbox->set_lbottomy(prior_pbox.lbottomy() + pbox.lbottomy());
			decode_pbox->set_rbottomx(prior_pbox.rbottomx() + pbox.rbottomx());
			decode_pbox->set_rbottomy(prior_pbox.rbottomy() + pbox.rbottomy());
			decode_pbox->set_rtopx(prior_pbox.rtopx() + pbox.rtopx());
			decode_pbox->set_rtopy(prior_pbox.rtopy() + pbox.rtopy());
			//decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
			//decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
			//decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
			//decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
		}
		else {
			// variance is encoded in pbox, we need to scale the offset accordingly.
			decode_pbox->set_ltopx(
				prior_pbox.ltopx() + prior_variance[0] * pbox.ltopx());
			decode_pbox->set_ltopy(
				prior_pbox.ltopy() + prior_variance[1] * pbox.ltopy());
			decode_pbox->set_lbottomx(
				prior_pbox.lbottomx() + prior_variance[2] * pbox.lbottomx());
			decode_pbox->set_lbottomy(
				prior_pbox.lbottomy() + prior_variance[3] * pbox.lbottomy());
			decode_pbox->set_rbottomx(
				prior_pbox.rbottomx() + prior_variance[4] * pbox.rbottomx());
			decode_pbox->set_rbottomy(
				prior_pbox.rbottomy() + prior_variance[5] * pbox.rbottomy());
			decode_pbox->set_rtopx(
				prior_pbox.rtopx() + prior_variance[6] * pbox.rtopx());
			decode_pbox->set_rtopy(
				prior_pbox.rtopy() + prior_variance[7] * pbox.rtopy());

		}
	}
	//*************************************************************************
	//else if (code_type == PriorPBoxParameter_CodeType_CENTER_SIZE) {
	//	float ltbcx = (prior_pbox.ltopx() + prior_pbox.lbottomx()) / 2.;
	//	float ltbcy = (prior_pbox.ltopy() + prior_pbox.lbottomy()) / 2.;
	//	float rtbcx = (prior_pbox.rtopx() + prior_pbox.rbottomx()) / 2.;
	//	float rtbcy = (prior_pbox.rtopy() + prior_pbox.rbottomy()) / 2.;
	//	float prior_width = sqrt(pow((ltbcx - rtbcx), 2.) + pow((ltbcy - rtbcy), 2.);
	//	CHECK_GT(prior_width, 0);
	//	float tlrcx = (prior_pbox.ltopx() + prior_pbox.rtopx()) / 2.;
	//	float tlrcy = (prior_pbox.ltopy() + prior_pbox.rtopy()) / 2.;
	//	float blrcx = (prior_pbox.lbottomx() + prior_pbox.rbottomx()) / 2.;
	//	float blrcy = (prior_pbox.lbottomy() + prior_pbox.rbottomy()) / 2.;
	//	float prior_height = sqrt(pow((tlrcx - blrcx), 2.) + pow((tlrcy - blrcy), 2.));
	//	CHECK_GT(prior_height, 0);
	//	float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
	//	float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

	//	float decode_bbox_center_x, decode_bbox_center_y;
	//	float decode_bbox_width, decode_bbox_height;
	//	if (variance_encoded_in_target) {
	//		// variance is encoded in target, we simply need to retore the offset
	//		// predictions.
	//		decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
	//		decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
	//		decode_bbox_width = exp(bbox.xmax()) * prior_width;
	//		decode_bbox_height = exp(bbox.ymax()) * prior_height;
	//	}
	//	else {
	//		// variance is encoded in bbox, we need to scale the offset accordingly.
	//		decode_bbox_center_x =
	//			prior_variance[0] * bbox.xmin() * prior_width + prior_center_x;
	//		decode_bbox_center_y =
	//			prior_variance[1] * bbox.ymin() * prior_height + prior_center_y;
	//		decode_bbox_width =
	//			exp(prior_variance[2] * bbox.xmax()) * prior_width;
	//		decode_bbox_height =
	//			exp(prior_variance[3] * bbox.ymax()) * prior_height;
	//	}

	//	decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
	//	decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
	//	decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
	//	decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
	//}
	//else if (code_type == PriorPBoxParameter_CodeType_CORNER_SIZE) {
	//	float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
	//	CHECK_GT(prior_width, 0);
	//	float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
	//	CHECK_GT(prior_height, 0);
	//	if (variance_encoded_in_target) {
	//		// variance is encoded in target, we simply need to add the offset
	//		// predictions.
	//		decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin() * prior_width);
	//		decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin() * prior_height);
	//		decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax() * prior_width);
	//		decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax() * prior_height);
	//	}
	//	else {
	//		// variance is encoded in bbox, we need to scale the offset accordingly.
	//		decode_bbox->set_xmin(
	//			prior_bbox.xmin() + prior_variance[0] * bbox.xmin() * prior_width);
	//		decode_bbox->set_ymin(
	//			prior_bbox.ymin() + prior_variance[1] * bbox.ymin() * prior_height);
	//		decode_bbox->set_xmax(
	//			prior_bbox.xmax() + prior_variance[2] * bbox.xmax() * prior_width);
	//		decode_bbox->set_ymax(
	//			prior_bbox.ymax() + prior_variance[3] * bbox.ymax() * prior_height);
	//	}
	//}
	else {
		LOG(FATAL) << "Unknown LocLossType.";
	}
	float pbox_size = PBoxSize(*decode_pbox);
	decode_pbox->set_size(pbox_size);
	if (clip_pbox) {
		ClipPBox(*decode_pbox, decode_pbox);
	}
}

void DecodePBoxes(
    const vector<NormalizedPBox>& prior_pboxes,
    const vector<vector<float> >& prior_variances,
    const PCodeType code_type, const bool variance_encoded_in_target,
    const bool clip_pbox, const vector<NormalizedPBox>& pboxes,
    vector<NormalizedPBox>* decode_pboxes) {
  CHECK_EQ(prior_pboxes.size(), prior_variances.size());
  CHECK_EQ(prior_pboxes.size(), pboxes.size());
  int num_pboxes = prior_pboxes.size();
  if (num_pboxes >= 1) {
    CHECK_EQ(prior_variances[0].size(), 8);
  }
  decode_pboxes->clear();
  for (int i = 0; i < num_pboxes; ++i) {
    NormalizedPBox decode_pbox;
    DecodePBox(prior_pboxes[i], prior_variances[i], code_type,
               variance_encoded_in_target, clip_pbox, pboxes[i], &decode_pbox);
    decode_pboxes->push_back(decode_pbox);
  }
}

void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const CodeType code_type, const bool variance_encoded_in_target,
    const bool clip, vector<LabelBBox>* all_decode_bboxes) {
  CHECK_EQ(all_loc_preds.size(), num);
  all_decode_bboxes->clear();
  all_decode_bboxes->resize(num);
  for (int i = 0; i < num; ++i) {
    // Decode predictions into bboxes.
    LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
    for (int c = 0; c < num_loc_classes; ++c) {
      int label = share_location ? -1 : c;
      if (label == background_label_id) {
        // Ignore background class.
        continue;
      }
      if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
        // Something bad happened if there are no predictions for current label.
        LOG(FATAL) << "Could not find location predictions for label " << label;
      }
      const vector<NormalizedBBox>& label_loc_preds =
          all_loc_preds[i].find(label)->second;
      DecodeBBoxes(prior_bboxes, prior_variances,
                   code_type, variance_encoded_in_target, clip,
                   label_loc_preds, &(decode_bboxes[label]));
    }
  }
}

void DecodePBoxesAll(const vector<LabelPBox>& all_loc_preds,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const int num, const bool share_location,
	const int num_loc_classes, const int background_label_id,
	const PCodeType code_type, const bool variance_encoded_in_target,
	const bool clip, vector<LabelPBox>* all_decode_pboxes) {
	CHECK_EQ(all_loc_preds.size(), num);
	all_decode_pboxes->clear();
	all_decode_pboxes->resize(num);
	for (int i = 0; i < num; ++i) {
		// Decode predictions into bboxes.
		LabelPBox& decode_pboxes = (*all_decode_pboxes)[i];
		for (int c = 0; c < num_loc_classes; ++c) {
			int label = share_location ? -1 : c;
			if (label == background_label_id) {
				// Ignore background class.
				continue;
			}
			if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label " << label;
			}
			const vector<NormalizedPBox>& label_loc_preds =
				all_loc_preds[i].find(label)->second;
			DecodePBoxes(prior_pboxes, prior_variances,
				code_type, variance_encoded_in_target, clip,
				label_loc_preds, &(decode_pboxes[label]));
		}
	}
}


void MatchBBox(const vector<NormalizedBBox>& gt_bboxes,
    const vector<NormalizedBBox>& pred_bboxes, const int label,
    const MatchType match_type, const float overlap_threshold,
    const bool ignore_cross_boundary_bbox,
    vector<int>* match_indices, vector<float>* match_overlaps) {
  int num_pred = pred_bboxes.size();
  match_indices->clear();
  match_indices->resize(num_pred, -1);
  match_overlaps->clear();
  match_overlaps->resize(num_pred, 0.);

  int num_gt = 0;
  vector<int> gt_indices;
  if (label == -1) {
    // label -1 means comparing against all ground truth.
    num_gt = gt_bboxes.size();
    for (int i = 0; i < num_gt; ++i) {
      gt_indices.push_back(i);
    }
  } else {
    // Count number of ground truth boxes which has the desired label.
    for (int i = 0; i < gt_bboxes.size(); ++i) {
      if (gt_bboxes[i].label() == label) {
        num_gt++;
        gt_indices.push_back(i);
      }
    }
  }
  if (num_gt == 0) {
    return;
  }

  // Store the positive overlap between predictions and ground truth.
  map<int, map<int, float> > overlaps;
  for (int i = 0; i < num_pred; ++i) {
    if (ignore_cross_boundary_bbox && IsCrossBoundaryBBox(pred_bboxes[i])) {
      (*match_indices)[i] = -2;
      continue;
    }
    for (int j = 0; j < num_gt; ++j) {
      float overlap = JaccardOverlap(pred_bboxes[i], gt_bboxes[gt_indices[j]]);
      if (overlap > 1e-6) {
        (*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
        overlaps[i][j] = overlap;
      }
    }
  }

  // Bipartite matching.
  vector<int> gt_pool;
  for (int i = 0; i < num_gt; ++i) {
    gt_pool.push_back(i);
  }
  while (gt_pool.size() > 0) {
    // Find the most overlapped gt and cooresponding predictions.
    int max_idx = -1;
    int max_gt_idx = -1;
    float max_overlap = -1;
    for (map<int, map<int, float> >::iterator it = overlaps.begin();
         it != overlaps.end(); ++it) {
      int i = it->first;
      if ((*match_indices)[i] != -1) {
        // The prediction already has matched ground truth or is ignored.
        continue;
      }
      for (int p = 0; p < gt_pool.size(); ++p) {
        int j = gt_pool[p];
        if (it->second.find(j) == it->second.end()) {
          // No overlap between the i-th prediction and j-th ground truth.
          continue;
        }
        // Find the maximum overlapped pair.
        if (it->second[j] > max_overlap) {
          // If the prediction has not been matched to any ground truth,
          // and the overlap is larger than maximum overlap, update.
          max_idx = i;
          max_gt_idx = j;
          max_overlap = it->second[j];
        }
      }
    }
    if (max_idx == -1) {
      // Cannot find good match.
      break;
    } else {
      CHECK_EQ((*match_indices)[max_idx], -1);
      (*match_indices)[max_idx] = gt_indices[max_gt_idx];
      (*match_overlaps)[max_idx] = max_overlap;
      // Erase the ground truth.
      gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
    }
  }

  switch (match_type) {
    case MultiBoxLossParameter_MatchType_BIPARTITE:
      // Already done.
      break;
    case MultiBoxLossParameter_MatchType_PER_PREDICTION:
      // Get most overlaped for the rest prediction bboxes.
      for (map<int, map<int, float> >::iterator it = overlaps.begin();
           it != overlaps.end(); ++it) {
        int i = it->first;
        if ((*match_indices)[i] != -1) {
          // The prediction already has matched ground truth or is ignored.
          continue;
        }
        int max_gt_idx = -1;
        float max_overlap = -1;
        for (int j = 0; j < num_gt; ++j) {
          if (it->second.find(j) == it->second.end()) {
            // No overlap between the i-th prediction and j-th ground truth.
            continue;
          }
          // Find the maximum overlapped pair.
          float overlap = it->second[j];
          if (overlap >= overlap_threshold && overlap > max_overlap) {
            // If the prediction has not been matched to any ground truth,
            // and the overlap is larger than maximum overlap, update.
            max_gt_idx = j;
            max_overlap = overlap;
          }
        }
        if (max_gt_idx != -1) {
          // Found a matched ground truth.
          CHECK_EQ((*match_indices)[i], -1);
          (*match_indices)[i] = gt_indices[max_gt_idx];
          (*match_overlaps)[i] = max_overlap;
        }
      }
      break;
    default:
      LOG(FATAL) << "Unknown matching type.";
      break;
  }

  return;
}

//******************************************************************
void MatchPBox(const vector<NormalizedPBox>& gt_pboxes,
	const vector<NormalizedPBox>& pred_pboxes, const int label,
	const PMatchType match_type, const float overlap_threshold,
	const bool ignore_cross_boundary_pbox,
	vector<int>* match_indices, vector<float>* match_overlaps) {
	int num_pred = pred_pboxes.size();
	match_indices->clear();
	match_indices->resize(num_pred, -1);
	match_overlaps->clear();
	match_overlaps->resize(num_pred, 0.);

	int num_gt = 0;
	vector<int> gt_indices;
	if (label == -1) {
		// label -1 means comparing against all ground truth.
		num_gt = gt_pboxes.size();
		for (int i = 0; i < num_gt; ++i) {
			gt_indices.push_back(i);
		}
	}
	else {
		// Count number of ground truth boxes which has the desired label.
		for (int i = 0; i < gt_pboxes.size(); ++i) {
			if (gt_pboxes[i].label() == label) {
				num_gt++;
				gt_indices.push_back(i);
			}
		}
	}
	if (num_gt == 0) {
		return;
	}

	// Store the positive overlap between predictions and ground truth.
	map<int, map<int, float> > overlaps;
	CHECK_GT(num_pred,0);
	for (int i = 0; i < num_pred; ++i) {
		//*************************************************************************
		if (ignore_cross_boundary_pbox && IsCrossBoundaryPBox(pred_pboxes[i])) {
			(*match_indices)[i] = -2;
			continue;
		}
		for (int j = 0; j < num_gt; ++j) {
			//LOG(INFO)<< "num_gt >0";
			//***********************************************************************
			float overlap = JaccardOverlapPbox(pred_pboxes[i], gt_pboxes[gt_indices[j]]);
			//LOG(INFO) << "overlap : " << overlap;
			if (overlap > 1e-6) {
				//LOG(FATAL) << "overlap > 1e-6";
				(*match_overlaps)[i] = std::max((*match_overlaps)[i], overlap);
				overlaps[i][j] = overlap;
			}
		}
	}

	// Bipartite matching.
	vector<int> gt_pool;
	for (int i = 0; i < num_gt; ++i) {
		gt_pool.push_back(i);
	}
	while (gt_pool.size() > 0) {
		// Find the most overlapped gt and cooresponding predictions.
		int max_idx = -1;
		int max_gt_idx = -1;
		float max_overlap = -1;
		for (map<int, map<int, float> >::iterator it = overlaps.begin();
			it != overlaps.end(); ++it) {
			int i = it->first;
			if ((*match_indices)[i] != -1) {
				// The prediction already has matched ground truth or is ignored.
				continue;
			}
			for (int p = 0; p < gt_pool.size(); ++p) {
				int j = gt_pool[p];
				if (it->second.find(j) == it->second.end()) {
					// No overlap between the i-th prediction and j-th ground truth.
					continue;
				}
				// Find the maximum overlapped pair.
				if (it->second[j] > max_overlap) {
					// If the prediction has not been matched to any ground truth,
					// and the overlap is larger than maximum overlap, update.
					max_idx = i;
					max_gt_idx = j;
					max_overlap = it->second[j];
				}
			}
		}
		if (max_idx == -1) {
			// Cannot find good match.
			break;
		}
		else {
			CHECK_EQ((*match_indices)[max_idx], -1);
			(*match_indices)[max_idx] = gt_indices[max_gt_idx];
			(*match_overlaps)[max_idx] = max_overlap;
			// Erase the ground truth.
			gt_pool.erase(std::find(gt_pool.begin(), gt_pool.end(), max_gt_idx));
		}
	}

	switch (match_type) {
	case MultiPBoxLossParameter_MatchType_BIPARTITE:
		// Already done.
		break;
	case MultiPBoxLossParameter_MatchType_PER_PREDICTION:
		// Get most overlaped for the rest prediction bboxes.
		for (map<int, map<int, float> >::iterator it = overlaps.begin();
			it != overlaps.end(); ++it) {
			int i = it->first;
			if ((*match_indices)[i] != -1) {
				// The prediction already has matched ground truth or is ignored.
				continue;
			}
			int max_gt_idx = -1;
			float max_overlap = -1;
			for (int j = 0; j < num_gt; ++j) {
				if (it->second.find(j) == it->second.end()) {
					// No overlap between the i-th prediction and j-th ground truth.
					continue;
				}
				// Find the maximum overlapped pair.
				float overlap = it->second[j];
				if (overlap >= overlap_threshold && overlap > max_overlap) {
					//LOG(FATAL) << "success";
					// If the prediction has not been matched to any ground truth,
					// and the overlap is larger than maximum overlap, update.
					max_gt_idx = j;
					max_overlap = overlap;
				}
			}
			if (max_gt_idx != -1) {
				// Found a matched ground truth.
				CHECK_EQ((*match_indices)[i], -1);
				(*match_indices)[i] = gt_indices[max_gt_idx];
				(*match_overlaps)[i] = max_overlap;
			}
		}
		break;
	default:
		LOG(FATAL) << "Unknown matching type.";
		break;
	}

	return;
}

void FindMatches(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      vector<map<int, vector<float> > >* all_match_overlaps,
      vector<map<int, vector<int> > >* all_match_indices) {
  // all_match_overlaps->clear();
  // all_match_indices->clear();
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const bool share_location = multibox_loss_param.share_location();
  const int loc_classes = share_location ? 1 : num_classes;
  const MatchType match_type = multibox_loss_param.match_type();
  const float overlap_threshold = multibox_loss_param.overlap_threshold();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  const int background_label_id = multibox_loss_param.background_label_id();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool ignore_cross_boundary_bbox =
      multibox_loss_param.ignore_cross_boundary_bbox();
  // Find the matches.
  int num = all_loc_preds.size();
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > match_indices;
    map<int, vector<float> > match_overlaps;
    // Check if there is ground truth for current image.
    if (all_gt_bboxes.find(i) == all_gt_bboxes.end()) {
      // There is no gt for current image. All predictions are negative.
      all_match_indices->push_back(match_indices);
      all_match_overlaps->push_back(match_overlaps);
      continue;
    }
    // Find match between predictions and ground truth.
    const vector<NormalizedBBox>& gt_bboxes = all_gt_bboxes.find(i)->second;
    if (!use_prior_for_matching) {
      for (int c = 0; c < loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (!share_location && label == background_label_id) {
          // Ignore background loc predictions.
          continue;
        }
        // Decode the prediction into bbox first.
        vector<NormalizedBBox> loc_bboxes;
        bool clip_bbox = false;
        DecodeBBoxes(prior_bboxes, prior_variances,
                     code_type, encode_variance_in_target, clip_bbox,
                     all_loc_preds[i].find(label)->second, &loc_bboxes);
        MatchBBox(gt_bboxes, loc_bboxes, label, match_type,
                  overlap_threshold, ignore_cross_boundary_bbox,
                  &match_indices[label], &match_overlaps[label]);
      }
    } else {
      // Use prior bboxes to match against all ground truth.
      vector<int> temp_match_indices;
      vector<float> temp_match_overlaps;
      const int label = -1;
      MatchBBox(gt_bboxes, prior_bboxes, label, match_type, overlap_threshold,
                ignore_cross_boundary_bbox, &temp_match_indices,
                &temp_match_overlaps);
      if (share_location) {
        match_indices[label] = temp_match_indices;
        match_overlaps[label] = temp_match_overlaps;
      } else {
        // Get ground truth label for each ground truth bbox.
        vector<int> gt_labels;
        for (int g = 0; g < gt_bboxes.size(); ++g) {
          gt_labels.push_back(gt_bboxes[g].label());
        }
        // Distribute the matching results to different loc_class.
        for (int c = 0; c < loc_classes; ++c) {
          if (c == background_label_id) {
            // Ignore background loc predictions.
            continue;
          }
          match_indices[c].resize(temp_match_indices.size(), -1);
          match_overlaps[c] = temp_match_overlaps;
          for (int m = 0; m < temp_match_indices.size(); ++m) {
            if (temp_match_indices[m] > -1) {
              const int gt_idx = temp_match_indices[m];
              CHECK_LT(gt_idx, gt_labels.size());
              if (c == gt_labels[gt_idx]) {
                match_indices[c][m] = gt_idx;
              }
            }
          }
        }
      }
    }
    all_match_indices->push_back(match_indices);
    all_match_overlaps->push_back(match_overlaps);
  }
}

//****************************************************************************
void FindPboxMatches(const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const MultiPBoxLossParameter& multipbox_loss_param,
	vector<map<int, vector<float> > >* all_match_overlaps,
	vector<map<int, vector<int> > >* all_match_indices) {
	// all_match_overlaps->clear();
	// all_match_indices->clear();
	// Get parameters.
	CHECK(multipbox_loss_param.has_num_classes()) << "Must provide num_classes.";
	const int num_classes = multipbox_loss_param.num_classes();
	CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
	const bool share_location = multipbox_loss_param.share_location();
	const int loc_classes = share_location ? 1 : num_classes;
	const PMatchType match_type = multipbox_loss_param.match_type();
	const float overlap_threshold = multipbox_loss_param.overlap_threshold();
	const bool use_prior_for_matching =
		multipbox_loss_param.use_prior_for_matching();
	const int background_label_id = multipbox_loss_param.background_label_id();
	const PCodeType code_type = multipbox_loss_param.code_type();
	const bool encode_variance_in_target =
		multipbox_loss_param.encode_variance_in_target();
	const bool ignore_cross_boundary_pbox =
		multipbox_loss_param.ignore_cross_boundary_pbox();
	// Find the matches.
	int num = all_loc_preds.size();
	for (int i = 0; i < num; ++i) {
		//LOG(INFO) << "num: " << num; 
		map<int, vector<int> > match_indices;
		map<int, vector<float> > match_overlaps;
		// Check if there is ground truth for current image.
		if (all_gt_pboxes.find(i) == all_gt_pboxes.end()) {
			//LOG(INFO) << "no gt!!!";
			// There is no gt for current image. All predictions are negative.
			all_match_indices->push_back(match_indices);
			all_match_overlaps->push_back(match_overlaps);
			continue;
		}
		// Find match between predictions and ground truth.
		const vector<NormalizedPBox>& gt_pboxes = all_gt_pboxes.find(i)->second;
		if (!use_prior_for_matching) {
			for (int c = 0; c < loc_classes; ++c) {
				int label = share_location ? -1 : c;
				if (!share_location && label == background_label_id) {
					// Ignore background loc predictions.
					continue;
				}
				// Decode the prediction into pbox first.
				vector<NormalizedPBox> loc_pboxes;
				bool clip_pbox = false;
				//*****************************************************
				DecodePBoxes(prior_pboxes, prior_variances,
					code_type, encode_variance_in_target, clip_pbox,
					all_loc_preds[i].find(label)->second, &loc_pboxes);

				MatchPBox(gt_pboxes, loc_pboxes, label, match_type,
					overlap_threshold, ignore_cross_boundary_pbox,
					&match_indices[label], &match_overlaps[label]);

				/*DecodeBBoxes(prior_bboxes, prior_variances,
					code_type, encode_variance_in_target, clip_bbox,
					all_loc_preds[i].find(label)->second, &loc_bboxes);*/

				  /*MatchBBox(gt_bboxes, loc_bboxes, label, match_type,
					overlap_threshold, ignore_cross_boundary_bbox,
					&match_indices[label], &match_overlaps[label]);*/
			}
		}
		else {
			//LOG(WARNING) << "using prior matching!";
			// Use prior pboxes to match against all ground truth.
			vector<int> temp_match_indices;
			vector<float> temp_match_overlaps;
			const int label = -1;
			//************************************************************************
			MatchPBox(gt_pboxes, prior_pboxes, label, match_type, overlap_threshold,
				ignore_cross_boundary_pbox, &temp_match_indices,
				&temp_match_overlaps);

			/*MatchBBox(gt_bboxes, prior_bboxes, label, match_type, overlap_threshold,
				ignore_cross_boundary_bbox, &temp_match_indices,
				&temp_match_overlaps);*/
			if (share_location) {
				match_indices[label] = temp_match_indices;
				match_overlaps[label] = temp_match_overlaps;
			}
			else {
				// Get ground truth label for each ground truth bbox.
				vector<int> gt_labels;
				for (int g = 0; g < gt_pboxes.size(); ++g) {
					gt_labels.push_back(gt_pboxes[g].label());
				}
				// Distribute the matching results to different loc_class.
				for (int c = 0; c < loc_classes; ++c) {
					if (c == background_label_id) {
						// Ignore background loc predictions.
						continue;
					}
					match_indices[c].resize(temp_match_indices.size(), -1);
					match_overlaps[c] = temp_match_overlaps;
					for (int m = 0; m < temp_match_indices.size(); ++m) {
						if (temp_match_indices[m] > -1) {
							const int gt_idx = temp_match_indices[m];
							CHECK_LT(gt_idx, gt_labels.size());
							if (c == gt_labels[gt_idx]) {
								match_indices[c][m] = gt_idx;
							}
						}
					}
				}
			}
		}
		all_match_indices->push_back(match_indices);
		all_match_overlaps->push_back(match_overlaps);
	}
}


int CountNumMatches(const vector<map<int, vector<int> > >& all_match_indices,
                    const int num) {
  int num_matches = 0;
  for (int i = 0; i < num; ++i) {
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (map<int, vector<int> >::const_iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const vector<int>& match_index = it->second;
      for (int m = 0; m < match_index.size(); ++m) {
        if (match_index[m] > -1) {
          ++num_matches;
        }
      }
    }
  }
  return num_matches;
}

inline bool IsEligibleMining(const MiningType mining_type, const int match_idx,
    const float match_overlap, const float neg_overlap) {
  if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
    return match_idx == -1 && match_overlap < neg_overlap;
  } else if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    return true;
  } else {
    return false;
  }
}
//pbox version
inline bool IsEligibleMining(const PMiningType mining_type, const int match_idx,
	const float match_overlap, const float neg_overlap) {
	if (mining_type == MultiPBoxLossParameter_MiningType_MAX_NEGATIVE) {
		return match_idx == -1 && match_overlap < neg_overlap;
	}
	else if (mining_type == MultiPBoxLossParameter_MiningType_HARD_EXAMPLE) {
		return true;
	}
	else {
		return false;
	}
}

template <typename Dtype>
void MineHardExamples(const Blob<Dtype>& conf_blob,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_overlaps.size());
  // CHECK_EQ(num, all_match_indices->size());
  // all_neg_indices->clear();
  *num_matches = CountNumMatches(*all_match_indices, num);
  *num_negs = 0;
  int num_priors = prior_bboxes.size();
  CHECK_EQ(num_priors, prior_variances.size());
  // Get parameters.
  CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
  const int num_classes = multibox_loss_param.num_classes();
  CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
  const int background_label_id = multibox_loss_param.background_label_id();
  const bool use_prior_for_nms = multibox_loss_param.use_prior_for_nms();
  const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
  const MiningType mining_type = multibox_loss_param.mining_type();
  if (mining_type == MultiBoxLossParameter_MiningType_NONE) {
    return;
  }
  const LocLossType loc_loss_type = multibox_loss_param.loc_loss_type();
  const float neg_pos_ratio = multibox_loss_param.neg_pos_ratio();
  const float neg_overlap = multibox_loss_param.neg_overlap();
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool has_nms_param = multibox_loss_param.has_nms_param();
  float nms_threshold = 0;
  int top_k = -1;
  if (has_nms_param) {
    nms_threshold = multibox_loss_param.nms_param().nms_threshold();
    top_k = multibox_loss_param.nms_param().top_k();
  }
  const int sample_size = multibox_loss_param.sample_size();
  // Compute confidence losses based on matching results.
  vector<vector<float> > all_conf_loss;
#ifdef CPU_ONLY
  ComputeConfLoss(conf_blob.cpu_data(), num, num_priors, num_classes,
      background_label_id, conf_loss_type, *all_match_indices, all_gt_bboxes,
      &all_conf_loss);
#else
  ComputeConfLossGPU(conf_blob, num, num_priors, num_classes,
      background_label_id, conf_loss_type, *all_match_indices, all_gt_bboxes,
      &all_conf_loss);
#endif
  vector<vector<float> > all_loc_loss;
  if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
    // Compute localization losses based on matching results.
    Blob<Dtype> loc_pred, loc_gt;
    if (*num_matches != 0) {
      vector<int> loc_shape(2, 1);
      loc_shape[1] = *num_matches * 4;
      loc_pred.Reshape(loc_shape);
      loc_gt.Reshape(loc_shape);
      Dtype* loc_pred_data = loc_pred.mutable_cpu_data();
      Dtype* loc_gt_data = loc_gt.mutable_cpu_data();
      EncodeLocPrediction(all_loc_preds, all_gt_bboxes, *all_match_indices,
                          prior_bboxes, prior_variances, multibox_loss_param,
                          loc_pred_data, loc_gt_data);
    }
    ComputeLocLoss(loc_pred, loc_gt, *all_match_indices, num,
                   num_priors, loc_loss_type, &all_loc_loss);
  } else {
    // No localization loss.
    for (int i = 0; i < num; ++i) {
      vector<float> loc_loss(num_priors, 0.f);
      all_loc_loss.push_back(loc_loss);
    }
  }
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> >& match_indices = (*all_match_indices)[i];
    const map<int, vector<float> >& match_overlaps = all_match_overlaps[i];
    // loc + conf loss.
    const vector<float>& conf_loss = all_conf_loss[i];
    const vector<float>& loc_loss = all_loc_loss[i];
    vector<float> loss;
    std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(),
                   std::back_inserter(loss), std::plus<float>());
    // Pick negatives or hard examples based on loss.
    set<int> sel_indices;
    vector<int> neg_indices;
    for (map<int, vector<int> >::iterator it = match_indices.begin();
         it != match_indices.end(); ++it) {
      const int label = it->first;
      int num_sel = 0;
      // Get potential indices and loss pairs.
      vector<pair<float, int> > loss_indices;
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (IsEligibleMining(mining_type, match_indices[label][m],
            match_overlaps.find(label)->second[m], neg_overlap)) {
          loss_indices.push_back(std::make_pair(loss[m], m));
          ++num_sel;
        }
      }
      if (mining_type == MultiBoxLossParameter_MiningType_MAX_NEGATIVE) {
        int num_pos = 0;
        for (int m = 0; m < match_indices[label].size(); ++m) {
          if (match_indices[label][m] > -1) {
            ++num_pos;
          }
        }
        num_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), num_sel);
      } else if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE) {
        CHECK_GT(sample_size, 0);
        num_sel = std::min(sample_size, num_sel);
      }
      // Select samples.
      if (has_nms_param && nms_threshold > 0) {
        // Do nms before selecting samples.
        vector<float> sel_loss;
        vector<NormalizedBBox> sel_bboxes;
        if (use_prior_for_nms) {
          for (int m = 0; m < match_indices[label].size(); ++m) {
            if (IsEligibleMining(mining_type, match_indices[label][m],
                match_overlaps.find(label)->second[m], neg_overlap)) {
              sel_loss.push_back(loss[m]);
              sel_bboxes.push_back(prior_bboxes[m]);
            }
          }
        } else {
          // Decode the prediction into bbox first.
          vector<NormalizedBBox> loc_bboxes;
          bool clip_bbox = false;
          DecodeBBoxes(prior_bboxes, prior_variances,
                       code_type, encode_variance_in_target, clip_bbox,
                       all_loc_preds[i].find(label)->second, &loc_bboxes);
          for (int m = 0; m < match_indices[label].size(); ++m) {
            if (IsEligibleMining(mining_type, match_indices[label][m],
                match_overlaps.find(label)->second[m], neg_overlap)) {
              sel_loss.push_back(loss[m]);
              sel_bboxes.push_back(loc_bboxes[m]);
            }
          }
        }
        // Do non-maximum suppression based on the loss.
        vector<int> nms_indices;
        ApplyNMS(sel_bboxes, sel_loss, nms_threshold, top_k, &nms_indices);
        if (nms_indices.size() < num_sel) {
          LOG(INFO) << "not enough sample after nms: " << nms_indices.size();
        }
        // Pick top example indices after nms.
        num_sel = std::min(static_cast<int>(nms_indices.size()), num_sel);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[nms_indices[n]].second);
        }
      } else {
        // Pick top example indices based on loss.
        std::sort(loss_indices.begin(), loss_indices.end(),
                  SortScorePairDescend<int>);
        for (int n = 0; n < num_sel; ++n) {
          sel_indices.insert(loss_indices[n].second);
        }
      }
      // Update the match_indices and select neg_indices.
      for (int m = 0; m < match_indices[label].size(); ++m) {
        if (match_indices[label][m] > -1) {
          if (mining_type == MultiBoxLossParameter_MiningType_HARD_EXAMPLE &&
              sel_indices.find(m) == sel_indices.end()) {
            match_indices[label][m] = -1;
            *num_matches -= 1;
          }
        } else if (match_indices[label][m] == -1) {
          if (sel_indices.find(m) != sel_indices.end()) {
            neg_indices.push_back(m);
            *num_negs += 1;
          }
        }
      }
    }
    all_neg_indices->push_back(neg_indices);
  }
}

// Explicite initialization.
template void MineHardExamples(const Blob<float>& conf_blob,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);
template void MineHardExamples(const Blob<double>& conf_blob,
    const vector<LabelBBox>& all_loc_preds,
    const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
    const vector<NormalizedBBox>& prior_bboxes,
    const vector<vector<float> >& prior_variances,
    const vector<map<int, vector<float> > >& all_match_overlaps,
    const MultiBoxLossParameter& multibox_loss_param,
    int* num_matches, int* num_negs,
    vector<map<int, vector<int> > >* all_match_indices,
    vector<vector<int> >* all_neg_indices);

//*********************************************************************
template <typename Dtype>
void MineHardExamples(const Blob<Dtype>& conf_blob,
	const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const vector<map<int, vector<float> > >& all_match_overlaps,
	const MultiPBoxLossParameter& multipbox_loss_param,
	int* num_matches, int* num_negs,
	vector<map<int, vector<int> > >* all_match_indices,
	vector<vector<int> >* all_neg_indices) {
	int num = all_loc_preds.size();
	// CHECK_EQ(num, all_match_overlaps.size());
	// CHECK_EQ(num, all_match_indices->size());
	// all_neg_indices->clear();
	*num_matches = CountNumMatches(*all_match_indices, num);
	//LOG(INFO) << "min hard num_match :" << (*num_matches);
	*num_negs = 0;
	int num_priors = prior_pboxes.size();
	CHECK_EQ(num_priors, prior_variances.size());
	// Get parameters.
	CHECK(multipbox_loss_param.has_num_classes()) << "Must provide num_classes.";
	const int num_classes = multipbox_loss_param.num_classes();
	CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
	const int background_label_id = multipbox_loss_param.background_label_id();
	const bool use_prior_for_nms = multipbox_loss_param.use_prior_for_nms();
	const PConfLossType conf_loss_type = multipbox_loss_param.conf_loss_type();
	const PMiningType mining_type = multipbox_loss_param.mining_type();
	if (mining_type == MultiPBoxLossParameter_MiningType_NONE) {
		return;
	}
	const PLocLossType loc_loss_type = multipbox_loss_param.loc_loss_type();
	const float neg_pos_ratio = multipbox_loss_param.neg_pos_ratio();
	const float neg_overlap = multipbox_loss_param.neg_overlap();
	const PCodeType code_type = multipbox_loss_param.code_type();
	const bool encode_variance_in_target =
		multipbox_loss_param.encode_variance_in_target();
	const bool has_nms_param = multipbox_loss_param.has_nms_param();
	float nms_threshold = 0;
	int top_k = -1;
	if (has_nms_param) {
		nms_threshold = multipbox_loss_param.nms_param().nms_threshold();
		top_k = multipbox_loss_param.nms_param().top_k();
	}
	const int sample_size = multipbox_loss_param.sample_size();
	// Compute confidence losses based on matching results.
	vector<vector<float> > all_conf_loss;
#ifdef CPU_ONLY
	ComputePboxConfLoss(conf_blob.cpu_data(), num, num_priors, num_classes,
		background_label_id, conf_loss_type, *all_match_indices, all_gt_pboxes,
		&all_conf_loss);
#else
	ComputePboxConfLossGPU(conf_blob, num, num_priors, num_classes,
		background_label_id, conf_loss_type, *all_match_indices, all_gt_pboxes,
		&all_conf_loss);
#endif
	vector<vector<float> > all_loc_loss;
	if (mining_type == MultiPBoxLossParameter_MiningType_HARD_EXAMPLE) {
		// Compute localization losses based on matching results.
		Blob<Dtype> loc_pred, loc_gt;
		if (*num_matches != 0) {
			vector<int> loc_shape(2, 1);
			loc_shape[1] = *num_matches * 8;
			loc_pred.Reshape(loc_shape);
			loc_gt.Reshape(loc_shape);
			Dtype* loc_pred_data = loc_pred.mutable_cpu_data();
			Dtype* loc_gt_data = loc_gt.mutable_cpu_data();
			EncodePboxLocPrediction(all_loc_preds, all_gt_pboxes, *all_match_indices,
				prior_pboxes, prior_variances, multipbox_loss_param,
				loc_pred_data, loc_gt_data);
		}
		ComputePLocLoss(loc_pred, loc_gt, *all_match_indices, num,
			num_priors, loc_loss_type, &all_loc_loss);
	}
	else {
		// No localization loss.
		for (int i = 0; i < num; ++i) {
			vector<float> loc_loss(num_priors, 0.f);
			all_loc_loss.push_back(loc_loss);
		}
	}
	for (int i = 0; i < num; ++i) {
		map<int, vector<int> >& match_indices = (*all_match_indices)[i];
		const map<int, vector<float> >& match_overlaps = all_match_overlaps[i];
		// loc + conf loss.
		const vector<float>& conf_loss = all_conf_loss[i];
		const vector<float>& loc_loss = all_loc_loss[i];
		vector<float> loss;
		std::transform(conf_loss.begin(), conf_loss.end(), loc_loss.begin(),
			std::back_inserter(loss), std::plus<float>());
			//LOG(WARNING) << "loss : " << loss[0];
		// Pick negatives or hard examples based on loss.
		set<int> sel_indices;
		vector<int> neg_indices;
		for (map<int, vector<int> >::iterator it = match_indices.begin();
			it != match_indices.end(); ++it) {
			const int label = it->first;
			int num_sel = 0;
			// Get potential indices and loss pairs.
			vector<pair<float, int> > loss_indices;
			for (int m = 0; m < match_indices[label].size(); ++m) {
				if (IsEligibleMining(mining_type, match_indices[label][m],
					match_overlaps.find(label)->second[m], neg_overlap)) {
					loss_indices.push_back(std::make_pair(loss[m], m));
					++num_sel;
				}
			}
			if (mining_type == MultiPBoxLossParameter_MiningType_MAX_NEGATIVE) {
				int num_pos = 0;
				for (int m = 0; m < match_indices[label].size(); ++m) {
					if (match_indices[label][m] > -1) {
						++num_pos;
					}
				}
				num_sel = std::min(static_cast<int>(num_pos * neg_pos_ratio), num_sel);
			}
			else if (mining_type == MultiPBoxLossParameter_MiningType_HARD_EXAMPLE) {
				CHECK_GT(sample_size, 0);
				num_sel = std::min(sample_size, num_sel);
			}
			// Select samples.
			if (has_nms_param && nms_threshold > 0) {
				// Do nms before selecting samples.
				vector<float> sel_loss;
				vector<NormalizedPBox> sel_pboxes;
				if (use_prior_for_nms) {
					for (int m = 0; m < match_indices[label].size(); ++m) {
						if (IsEligibleMining(mining_type, match_indices[label][m],
							match_overlaps.find(label)->second[m], neg_overlap)) {
							sel_loss.push_back(loss[m]);
							sel_pboxes.push_back(prior_pboxes[m]);
						}
					}
				}
				else {
					// Decode the prediction into bbox first.
					vector<NormalizedPBox> loc_pboxes;
					bool clip_pbox = false;
					DecodePBoxes(prior_pboxes, prior_variances,
						code_type, encode_variance_in_target, clip_pbox,
						all_loc_preds[i].find(label)->second, &loc_pboxes);
					for (int m = 0; m < match_indices[label].size(); ++m) {
						if (IsEligibleMining(mining_type, match_indices[label][m],
							match_overlaps.find(label)->second[m], neg_overlap)) {
							sel_loss.push_back(loss[m]);
							sel_pboxes.push_back(loc_pboxes[m]);
						}
					}
				}
				// Do non-maximum suppression based on the loss.
				vector<int> nms_indices;
				ApplyPboxNMS(sel_pboxes, sel_loss, nms_threshold, top_k, &nms_indices);
				if (nms_indices.size() < num_sel) {
					LOG(INFO) << "not enough sample after nms: " << nms_indices.size();
				}
				// Pick top example indices after nms.
				num_sel = std::min(static_cast<int>(nms_indices.size()), num_sel);
				for (int n = 0; n < num_sel; ++n) {
					sel_indices.insert(loss_indices[nms_indices[n]].second);
				}
			}
			else {
				// Pick top example indices based on loss.
				std::sort(loss_indices.begin(), loss_indices.end(),
					SortScorePairDescend<int>);
				for (int n = 0; n < num_sel; ++n) {
					sel_indices.insert(loss_indices[n].second);
				}
			}
			// Update the match_indices and select neg_indices.
			for (int m = 0; m < match_indices[label].size(); ++m) {
				if (match_indices[label][m] > -1) {
					if (mining_type == MultiPBoxLossParameter_MiningType_HARD_EXAMPLE &&
						sel_indices.find(m) == sel_indices.end()) {
						match_indices[label][m] = -1;
						*num_matches -= 1;
					}
				}
				else if (match_indices[label][m] == -1) {
					if (sel_indices.find(m) != sel_indices.end()) {
						neg_indices.push_back(m);
						*num_negs += 1;
					}
				}
			}
		}
		all_neg_indices->push_back(neg_indices);
	}
}

// Explicite initialization.
template void MineHardExamples(const Blob<float>& conf_blob,
	const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const vector<map<int, vector<float> > >& all_match_overlaps,
	const MultiPBoxLossParameter& multipbox_loss_param,
	int* num_matches, int* num_negs,
	vector<map<int, vector<int> > >* all_match_indices,
	vector<vector<int> >* all_neg_indices);
template void MineHardExamples(const Blob<double>& conf_blob,
	const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const vector<map<int, vector<float> > >& all_match_overlaps,
	const MultiPBoxLossParameter& multipbox_loss_param,
	int* num_matches, int* num_negs,
	vector<map<int, vector<int> > >* all_match_indices,
	vector<vector<int> >* all_neg_indices);


template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    NormalizedBBox bbox;
    bbox.set_label(label);
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);
template void GetGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, vector<NormalizedBBox> >* all_gt_bboxes);

//************************************************************************
template <typename Dtype>
void GetPboxGroundTruth(const Dtype* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, vector<NormalizedPBox> >* all_gt_pboxes) {
	all_gt_pboxes->clear();
	for (int i = 0; i < num_gt; ++i) {
		int start_idx = i * 12;
		int item_id = gt_data[start_idx];
		//LOG(INFO) << "item id: " << item_id;
		if (item_id == -1) {
			continue;
		}
		int label = gt_data[start_idx + 1];
		CHECK_NE(background_label_id, label)
			<< "Found background label in the dataset.";
		bool difficult = static_cast<bool>(gt_data[start_idx + 11]);
		if (!use_difficult_gt && difficult) {
			// Skip reading difficult ground truth.
			continue;
		}
		NormalizedPBox pbox;
		pbox.set_label(label);
		pbox.set_ltopx(gt_data[start_idx + 3]);
		pbox.set_ltopy(gt_data[start_idx + 4]);
		pbox.set_lbottomx(gt_data[start_idx + 5]);
		pbox.set_lbottomy(gt_data[start_idx + 6]);
		pbox.set_rbottomx(gt_data[start_idx + 7]);
		pbox.set_rbottomy(gt_data[start_idx + 8]);
		pbox.set_rtopx(gt_data[start_idx + 9]);
		pbox.set_rtopy(gt_data[start_idx + 10]);
		pbox.set_difficult(difficult);
		float pbox_size = PBoxSize(pbox);
		//LOG(WARNING) << "GT PBOX size :" << pbox_size;
		pbox.set_size(pbox_size);
		(*all_gt_pboxes)[item_id].push_back(pbox);
	}
}

// Explicit initialization.
template void GetPboxGroundTruth(const float* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, vector<NormalizedPBox> >* all_gt_pboxes);
template void GetPboxGroundTruth(const double* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, vector<NormalizedPBox> >* all_gt_pboxes);

template <typename Dtype>
void GetGroundTruth(const Dtype* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes) {
  all_gt_bboxes->clear();
  for (int i = 0; i < num_gt; ++i) {
    int start_idx = i * 8;
    int item_id = gt_data[start_idx];
    if (item_id == -1) {
      break;
    }
    NormalizedBBox bbox;
    int label = gt_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the dataset.";
    bool difficult = static_cast<bool>(gt_data[start_idx + 7]);
    if (!use_difficult_gt && difficult) {
      // Skip reading difficult ground truth.
      continue;
    }
    bbox.set_xmin(gt_data[start_idx + 3]);
    bbox.set_ymin(gt_data[start_idx + 4]);
    bbox.set_xmax(gt_data[start_idx + 5]);
    bbox.set_ymax(gt_data[start_idx + 6]);
    bbox.set_difficult(difficult);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_gt_bboxes)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void GetGroundTruth(const float* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);
template void GetGroundTruth(const double* gt_data, const int num_gt,
      const int background_label_id, const bool use_difficult_gt,
      map<int, LabelBBox>* all_gt_bboxes);

//*********************************************************************
template <typename Dtype>
void GetPboxGroundTruth(const Dtype* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, LabelPBox>* all_gt_pboxes) {
	all_gt_pboxes->clear();
	for (int i = 0; i < num_gt; ++i) {
		int start_idx = i * 12;
		int item_id = gt_data[start_idx];
		if (item_id == -1) {
			break;
		}
		NormalizedPBox pbox;
		int label = gt_data[start_idx + 1];
		CHECK_NE(background_label_id, label)
			<< "Found background label in the dataset.";
		bool difficult = static_cast<bool>(gt_data[start_idx + 11]);
		if (!use_difficult_gt && difficult) {
			// Skip reading difficult ground truth.
			continue;
		}
		pbox.set_label(label);
		pbox.set_ltopx(gt_data[start_idx + 3]);
		pbox.set_ltopy(gt_data[start_idx + 4]);
		pbox.set_lbottomx(gt_data[start_idx + 5]);
		pbox.set_lbottomy(gt_data[start_idx + 6]);
		pbox.set_rbottomx(gt_data[start_idx + 7]);
		pbox.set_rbottomy(gt_data[start_idx + 8]);
		pbox.set_rtopx(gt_data[start_idx + 9]);
		pbox.set_rtopy(gt_data[start_idx + 10]);
		pbox.set_difficult(difficult);
		float pbox_size = PBoxSize(pbox);
		pbox.set_size(pbox_size);
		(*all_gt_pboxes)[item_id][label].push_back(pbox);
		/*bbox.set_xmin(gt_data[start_idx + 3]);
		bbox.set_ymin(gt_data[start_idx + 4]);
		bbox.set_xmax(gt_data[start_idx + 5]);
		bbox.set_ymax(gt_data[start_idx + 6]);
		bbox.set_difficult(difficult);
		float bbox_size = PBoxSize(pbox);
		bbox.set_size(bbox_size);
		(*all_gt_bboxes)[item_id][label].push_back(bbox);*/
	}
}

// Explicit initialization.
template void GetPboxGroundTruth(const float* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, LabelPBox>* all_gt_pboxes);
template void GetPboxGroundTruth(const double* gt_data, const int num_gt,
	const int background_label_id, const bool use_difficult_gt,
	map<int, LabelPBox>* all_gt_pboxes);

template <typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds) {
  loc_preds->clear();
  if (share_location) {
    CHECK_EQ(num_loc_classes, 1);
  }
  loc_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    LabelBBox& label_bbox = (*loc_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_loc_classes * 4;
      for (int c = 0; c < num_loc_classes; ++c) {
        int label = share_location ? -1 : c;
        if (label_bbox.find(label) == label_bbox.end()) {
          label_bbox[label].resize(num_preds_per_class);
        }
        label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
        label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
        label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
        label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
      }
    }
    loc_data += num_preds_per_class * num_loc_classes * 4;
  }
}

// Explicit initialization.
template void GetLocPredictions(const float* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);
template void GetLocPredictions(const double* loc_data, const int num,
      const int num_preds_per_class, const int num_loc_classes,
      const bool share_location, vector<LabelBBox>* loc_preds);

//************************************************************************
template <typename Dtype>
void GetPboxLocPredictions(const Dtype* loc_data, const int num,
	const int num_preds_per_class, const int num_loc_classes,
	const bool share_location, vector<LabelPBox>* loc_preds) {
	loc_preds->clear();
	if (share_location) {
		CHECK_EQ(num_loc_classes, 1);
	}
	loc_preds->resize(num);
	for (int i = 0; i < num; ++i) {
		LabelPBox& label_pbox = (*loc_preds)[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_loc_classes * 8;
			for (int c = 0; c < num_loc_classes; ++c) {
				int label = share_location ? -1 : c;
				if (label_pbox.find(label) == label_pbox.end()) {
					label_pbox[label].resize(num_preds_per_class);
				}
				label_pbox[label][p].set_ltopx(loc_data[start_idx + c * 8]);
				label_pbox[label][p].set_ltopy(loc_data[start_idx + c * 8 + 1]);
				label_pbox[label][p].set_lbottomx(loc_data[start_idx + c * 8 + 2]);
				label_pbox[label][p].set_lbottomy(loc_data[start_idx + c * 8 + 3]);
				label_pbox[label][p].set_rbottomx(loc_data[start_idx + c * 8 + 4]);
				label_pbox[label][p].set_rbottomy(loc_data[start_idx + c * 8 + 5]);
				label_pbox[label][p].set_rtopx(loc_data[start_idx + c * 8 + 6]);
				label_pbox[label][p].set_rtopy(loc_data[start_idx + c * 8 + 7]);
			}
		}
		loc_data += num_preds_per_class * num_loc_classes * 8;
	}
}

// Explicit initialization.
template void GetPboxLocPredictions(const float* loc_data, const int num,
	const int num_preds_per_class, const int num_loc_classes,
	const bool share_location, vector<LabelPBox>* loc_preds);
template void GetPboxLocPredictions(const double* loc_data, const int num,
	const int num_preds_per_class, const int num_loc_classes,
	const bool share_location, vector<LabelPBox>* loc_preds);

template <typename Dtype>
void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      Dtype* loc_pred_data, Dtype* loc_gt_data) {
  int num = all_loc_preds.size();
  // CHECK_EQ(num, all_match_indices.size());
  // Get parameters.
  const CodeType code_type = multibox_loss_param.code_type();
  const bool encode_variance_in_target =
      multibox_loss_param.encode_variance_in_target();
  const bool bp_inside = multibox_loss_param.bp_inside();
  const bool use_prior_for_matching =
      multibox_loss_param.use_prior_for_matching();
  int count = 0;
  for (int i = 0; i < num; ++i) {
    for (map<int, vector<int> >::const_iterator
         it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
      const int label = it->first;
      const vector<int>& match_index = it->second;
      CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
      const vector<NormalizedBBox>& loc_pred =
          all_loc_preds[i].find(label)->second;
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        // Store encoded ground truth.
        const int gt_idx = match_index[j];
        CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
        CHECK_LT(gt_idx, all_gt_bboxes.find(i)->second.size());
        const NormalizedBBox& gt_bbox = all_gt_bboxes.find(i)->second[gt_idx];
        NormalizedBBox gt_encode;
        CHECK_LT(j, prior_bboxes.size());
        EncodeBBox(prior_bboxes[j], prior_variances[j], code_type,
                   encode_variance_in_target, gt_bbox, &gt_encode);
        loc_gt_data[count * 4] = gt_encode.xmin();
        loc_gt_data[count * 4 + 1] = gt_encode.ymin();
        loc_gt_data[count * 4 + 2] = gt_encode.xmax();
        loc_gt_data[count * 4 + 3] = gt_encode.ymax();
        // Store location prediction.
        CHECK_LT(j, loc_pred.size());
        if (bp_inside) {
          NormalizedBBox match_bbox = prior_bboxes[j];
          if (!use_prior_for_matching) {
            const bool clip_bbox = false;
            DecodeBBox(prior_bboxes[j], prior_variances[j], code_type,
                       encode_variance_in_target, clip_bbox, loc_pred[j],
                       &match_bbox);
          }
          // When a dimension of match_bbox is outside of image region, use
          // gt_encode to simulate zero gradient.
          loc_pred_data[count * 4] =
              (match_bbox.xmin() < 0 || match_bbox.xmin() > 1) ?
              gt_encode.xmin() : loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] =
              (match_bbox.ymin() < 0 || match_bbox.ymin() > 1) ?
              gt_encode.ymin() : loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] =
              (match_bbox.xmax() < 0 || match_bbox.xmax() > 1) ?
              gt_encode.xmax() : loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] =
              (match_bbox.ymax() < 0 || match_bbox.ymax() > 1) ?
              gt_encode.ymax() : loc_pred[j].ymax();
        } else {
          loc_pred_data[count * 4] = loc_pred[j].xmin();
          loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
          loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
          loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
        }
        if (encode_variance_in_target) {
          for (int k = 0; k < 4; ++k) {
            CHECK_GT(prior_variances[j][k], 0);
            loc_pred_data[count * 4 + k] /= prior_variances[j][k];
            loc_gt_data[count * 4 + k] /= prior_variances[j][k];
          }
        }
        ++count;
      }
    }
  }
}

// Explicit initialization.
template void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      float* loc_pred_data, float* loc_gt_data);
template void EncodeLocPrediction(const vector<LabelBBox>& all_loc_preds,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<NormalizedBBox>& prior_bboxes,
      const vector<vector<float> >& prior_variances,
      const MultiBoxLossParameter& multibox_loss_param,
      double* loc_pred_data, double* loc_gt_data);

//************************************************************************
template <typename Dtype>
void EncodePboxLocPrediction(const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const MultiPBoxLossParameter& multipbox_loss_param,
	Dtype* loc_pred_data, Dtype* loc_gt_data) {
	int num = all_loc_preds.size();
	// CHECK_EQ(num, all_match_indices.size());
	// Get parameters.
	const PCodeType code_type = multipbox_loss_param.code_type();
	const bool encode_variance_in_target =
		multipbox_loss_param.encode_variance_in_target();
	const bool bp_inside = multipbox_loss_param.bp_inside();
	const bool use_prior_for_matching =
		multipbox_loss_param.use_prior_for_matching();
	int count = 0;
	for (int i = 0; i < num; ++i) {
		for (map<int, vector<int> >::const_iterator
			it = all_match_indices[i].begin();
			it != all_match_indices[i].end(); ++it) {
			const int label = it->first;
			const vector<int>& match_index = it->second;
			CHECK(all_loc_preds[i].find(label) != all_loc_preds[i].end());
			const vector<NormalizedPBox>& loc_pred =
				all_loc_preds[i].find(label)->second;
			for (int j = 0; j < match_index.size(); ++j) {
				if (match_index[j] <= -1) {
					continue;
				}
				// Store encoded ground truth.
				const int gt_idx = match_index[j];
				CHECK(all_gt_pboxes.find(i) != all_gt_pboxes.end());
				CHECK_LT(gt_idx, all_gt_pboxes.find(i)->second.size());
				const NormalizedPBox& gt_pbox = all_gt_pboxes.find(i)->second[gt_idx];
				NormalizedPBox gt_encode;
				CHECK_LT(j, prior_pboxes.size());
				//**********************************************************
				EncodePBox(prior_pboxes[j], prior_variances[j], code_type,
					encode_variance_in_target, gt_pbox, &gt_encode);

				loc_gt_data[count * 8] = gt_encode.ltopx();
				loc_gt_data[count * 8 + 1] = gt_encode.ltopy();
				loc_gt_data[count * 8 + 2] = gt_encode.lbottomx();
				loc_gt_data[count * 8 + 3] = gt_encode.lbottomy();
				loc_gt_data[count * 8 + 4] = gt_encode.rbottomx();
				loc_gt_data[count * 8 + 5] = gt_encode.rbottomy();
				loc_gt_data[count * 8 + 6] = gt_encode.rtopx();
				loc_gt_data[count * 8 + 7] = gt_encode.rtopy();
				// Store location prediction.
				//****************************** 
				CHECK_LT(j, loc_pred.size());
				if (bp_inside) {
					NormalizedPBox match_pbox = prior_pboxes[j];
					if (!use_prior_for_matching) {
						const bool clip_pbox = false;
						DecodePBox(prior_pboxes[j], prior_variances[j], code_type,
							encode_variance_in_target, clip_pbox, loc_pred[j],
							&match_pbox);
					}
					// When a dimension of match_bbox is outside of image region, use
					// gt_encode to simulate zero gradient.
					loc_pred_data[count * 8] =
						(match_pbox.ltopx() < 0 || match_pbox.ltopx() > 1) ?
						gt_encode.ltopx() : loc_pred[j].ltopx();
					loc_pred_data[count * 8 + 1] =
						(match_pbox.ltopy() < 0 || match_pbox.ltopy() > 1) ?
						gt_encode.ltopy() : loc_pred[j].ltopy();
					loc_pred_data[count * 8 + 2] =
						(match_pbox.lbottomx() < 0 || match_pbox.lbottomx() > 1) ?
						gt_encode.lbottomx() : loc_pred[j].lbottomx();
					loc_pred_data[count * 8 + 3] =
						(match_pbox.lbottomy() < 0 || match_pbox.lbottomy() > 1) ?
						gt_encode.lbottomy() : loc_pred[j].lbottomy();
					loc_pred_data[count * 8 + 4] =
						(match_pbox.rbottomx() < 0 || match_pbox.rbottomx() > 1) ?
						gt_encode.rbottomx() : loc_pred[j].rbottomx();
					loc_pred_data[count * 8 + 5] =
						(match_pbox.rbottomy() < 0 || match_pbox.rbottomy() > 1) ?
						gt_encode.rbottomy() : loc_pred[j].rbottomy();
					loc_pred_data[count * 8 + 6] =
						(match_pbox.rtopx() < 0 || match_pbox.rtopx() > 1) ?
						gt_encode.rtopx() : loc_pred[j].rtopx();
					loc_pred_data[count * 8 + 7] =
						(match_pbox.rtopy() < 0 || match_pbox.rtopy() > 1) ?
						gt_encode.rtopy() : loc_pred[j].rtopy();

					/*loc_pred_data[count * 4] =
						(match_pbox.xmin() < 0 || match_pbox.xmin() > 1) ?
						gt_encode.xmin() : loc_pred[j].xmin();
					loc_pred_data[count * 4 + 1] =
						(match_pbox.ymin() < 0 || match_pbox.ymin() > 1) ?
						gt_encode.ymin() : loc_pred[j].ymin();
					loc_pred_data[count * 4 + 2] =
						(match_bbox.xmax() < 0 || match_bbox.xmax() > 1) ?
						gt_encode.xmax() : loc_pred[j].xmax();
					loc_pred_data[count * 4 + 3] =
						(match_bbox.ymax() < 0 || match_bbox.ymax() > 1) ?
						gt_encode.ymax() : loc_pred[j].ymax();*/
				}
				else {
					loc_pred_data[count * 8] = loc_pred[j].ltopx();
					loc_pred_data[count * 8 + 1] = loc_pred[j].ltopy();
					loc_pred_data[count * 8 + 2] = loc_pred[j].lbottomx();
					loc_pred_data[count * 8 + 3] = loc_pred[j].lbottomy();
					loc_pred_data[count * 8 + 4] = loc_pred[j].rbottomx();
					loc_pred_data[count * 8 + 5] = loc_pred[j].rbottomy();
					loc_pred_data[count * 8 + 6] = loc_pred[j].rtopx();
					loc_pred_data[count * 8 + 7] = loc_pred[j].rtopy();

					//loc_pred_data[count * 4] = loc_pred[j].xmin();
					//loc_pred_data[count * 4 + 1] = loc_pred[j].ymin();
					//loc_pred_data[count * 4 + 2] = loc_pred[j].xmax();
					//loc_pred_data[count * 4 + 3] = loc_pred[j].ymax();
				}
				if (encode_variance_in_target) {
					for (int k = 0; k < 8; ++k) {
						CHECK_GT(prior_variances[j][k], 0);
						loc_pred_data[count * 8 + k] /= prior_variances[j][k];
						loc_gt_data[count * 8 + k] /= prior_variances[j][k];
					}
				}
				++count;
			}
		}
	}
}

// Explicit initialization.
template void EncodePboxLocPrediction(const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const MultiPBoxLossParameter& multipbox_loss_param,
	float* loc_pred_data, float* loc_gt_data);

template void EncodePboxLocPrediction(const vector<LabelPBox>& all_loc_preds,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<NormalizedPBox>& prior_pboxes,
	const vector<vector<float> >& prior_variances,
	const MultiPBoxLossParameter& multipbox_loss_param,
	double* loc_pred_data, double* loc_gt_data);

template <typename Dtype>
void ComputeLocLoss(const Blob<Dtype>& loc_pred, const Blob<Dtype>& loc_gt,
      const vector<map<int, vector<int> > >& all_match_indices,
      const int num, const int num_priors, const LocLossType loc_loss_type,
      vector<vector<float> >* all_loc_loss) {
  int loc_count = loc_pred.count();
  CHECK_EQ(loc_count, loc_gt.count());
  Blob<Dtype> diff;
  const Dtype* diff_data = NULL;
  if (loc_count != 0) {
    diff.Reshape(loc_pred.shape());
    caffe_sub(loc_count, loc_pred.cpu_data(), loc_gt.cpu_data(),
              diff.mutable_cpu_data());
    diff_data = diff.cpu_data();
  }
  int count = 0;
  for (int i = 0; i < num; ++i) {
    vector<float> loc_loss(num_priors, 0.f);
    for (map<int, vector<int> >::const_iterator
         it = all_match_indices[i].begin();
         it != all_match_indices[i].end(); ++it) {
      const vector<int>& match_index = it->second;
      CHECK_EQ(num_priors, match_index.size());
      for (int j = 0; j < match_index.size(); ++j) {
        if (match_index[j] <= -1) {
          continue;
        }
        Dtype loss = 0;
        for (int k = 0; k < 4; ++k) {
          Dtype val = diff_data[count * 4 + k];
          if (loc_loss_type == MultiBoxLossParameter_LocLossType_SMOOTH_L1) {
            Dtype abs_val = fabs(val);
            if (abs_val < 1.) {
              loss += 0.5 * val * val;
            } else {
              loss += abs_val - 0.5;
            }
          } else if (loc_loss_type == MultiBoxLossParameter_LocLossType_L2) {
            loss += 0.5 * val * val;
          } else {
            LOG(FATAL) << "Unknown loc loss type.";
          }
        }
        loc_loss[j] = loss;
        ++count;
      }
    }
    all_loc_loss->push_back(loc_loss);
  }
}

// Explicit initialization.
template void ComputeLocLoss(const Blob<float>& loc_pred,
      const Blob<float>& loc_gt,
      const vector<map<int, vector<int> > >& all_match_indices,
      const int num, const int num_priors, const LocLossType loc_loss_type,
      vector<vector<float> >* all_loc_loss);
template void ComputeLocLoss(const Blob<double>& loc_pred,
      const Blob<double>& loc_gt,
      const vector<map<int, vector<int> > >& all_match_indices,
      const int num, const int num_priors, const LocLossType loc_loss_type,
      vector<vector<float> >* all_loc_loss);


template <typename Dtype>
void ComputePLocLoss(const Blob<Dtype>& loc_pred, const Blob<Dtype>& loc_gt,
	const vector<map<int, vector<int> > >& all_match_indices,
	const int num, const int num_priors, const PLocLossType loc_loss_type,
	vector<vector<float> >* all_loc_loss) {
	int loc_count = loc_pred.count();
	CHECK_EQ(loc_count, loc_gt.count());
	Blob<Dtype> diff;
	const Dtype* diff_data = NULL;
	if (loc_count != 0) {
		diff.Reshape(loc_pred.shape());
		caffe_sub(loc_count, loc_pred.cpu_data(), loc_gt.cpu_data(),
			diff.mutable_cpu_data());
		diff_data = diff.cpu_data();
	}
	int count = 0;
	for (int i = 0; i < num; ++i) {
		vector<float> loc_loss(num_priors, 0.f);
		for (map<int, vector<int> >::const_iterator
			it = all_match_indices[i].begin();
			it != all_match_indices[i].end(); ++it) {
			const vector<int>& match_index = it->second;
			CHECK_EQ(num_priors, match_index.size());
			for (int j = 0; j < match_index.size(); ++j) {
				if (match_index[j] <= -1) {
					continue;
				}
				Dtype loss = 0;
				for (int k = 0; k < 8; ++k) {
					Dtype val = diff_data[count * 8 + k];
					if (loc_loss_type == MultiPBoxLossParameter_LocLossType_SMOOTH_L1) {
						Dtype abs_val = fabs(val);
						if (abs_val < 1.) {
							loss += 0.5 * val * val;
						}
						else {
							loss += abs_val - 0.5;
						}
					}
					else if (loc_loss_type == MultiPBoxLossParameter_LocLossType_L2) {
						loss += 0.5 * val * val;
					}
					else {
						LOG(FATAL) << "Unknown loc loss type.";
					}
				}
				loc_loss[j] = loss;
				++count;
			}
		}
		all_loc_loss->push_back(loc_loss);
	}
}

// Explicit initialization.
template void ComputePLocLoss(const Blob<float>& loc_pred,
	const Blob<float>& loc_gt,
	const vector<map<int, vector<int> > >& all_match_indices,
	const int num, const int num_priors, const PLocLossType loc_loss_type,
	vector<vector<float> >* all_loc_loss);
template void ComputePLocLoss(const Blob<double>& loc_pred,
	const Blob<double>& loc_gt,
	const vector<map<int, vector<int> > >& all_match_indices,
	const int num, const int num_priors, const PLocLossType loc_loss_type,
	vector<vector<float> >* all_loc_loss);


template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& label_scores = (*conf_preds)[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].push_back(conf_data[start_idx + c]);
      }
    }
    conf_data += num_preds_per_class * num_classes;
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds);
template void GetConfidenceScores(const double* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      vector<map<int, vector<float> > >* conf_preds);

template <typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const bool class_major, vector<map<int, vector<float> > >* conf_preds) {
  conf_preds->clear();
  conf_preds->resize(num);
  for (int i = 0; i < num; ++i) {
    map<int, vector<float> >& label_scores = (*conf_preds)[i];
    if (class_major) {
      for (int c = 0; c < num_classes; ++c) {
        label_scores[c].assign(conf_data, conf_data + num_preds_per_class);
        conf_data += num_preds_per_class;
      }
    } else {
      for (int p = 0; p < num_preds_per_class; ++p) {
        int start_idx = p * num_classes;
        for (int c = 0; c < num_classes; ++c) {
          label_scores[c].push_back(conf_data[start_idx + c]);
        }
      }
      conf_data += num_preds_per_class * num_classes;
    }
  }
}

// Explicit initialization.
template void GetConfidenceScores(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const bool class_major, vector<map<int, vector<float> > >* conf_preds);
template void GetConfidenceScores(const double* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const bool class_major, vector<map<int, vector<float> > >* conf_preds);

template <typename Dtype>
void ComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      vector<vector<float> >* all_conf_loss) {
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      int label = background_label_id;
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
              log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

// Explicit initialization.
template void ComputeConfLoss(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      vector<vector<float> >* all_conf_loss);
template void ComputeConfLoss(const double* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      vector<vector<float> >* all_conf_loss);

template <typename Dtype>
void ComputeConfLoss(const Dtype* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss) {
  CHECK_LT(background_label_id, num_classes);
  // CHECK_EQ(num, all_match_indices.size());
  all_conf_loss->clear();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss;
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
      int start_idx = p * num_classes;
      // Get the label index.
      int label = background_label_id;
      for (map<int, vector<int> >::const_iterator it =
           match_indices.begin(); it != match_indices.end(); ++it) {
        const vector<int>& match_index = it->second;
        CHECK_EQ(match_index.size(), num_preds_per_class);
        if (match_index[p] > -1) {
          CHECK(all_gt_bboxes.find(i) != all_gt_bboxes.end());
          const vector<NormalizedBBox>& gt_bboxes =
              all_gt_bboxes.find(i)->second;
          CHECK_LT(match_index[p], gt_bboxes.size());
          label = gt_bboxes[match_index[p]].label();
          CHECK_GE(label, 0);
          CHECK_NE(label, background_label_id);
          CHECK_LT(label, num_classes);
          // A prior can only be matched to one gt bbox.
          break;
        }
      }
      Dtype loss = 0;
      if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
        CHECK_GE(label, 0);
        CHECK_LT(label, num_classes);
        // Compute softmax probability.
        // We need to subtract the max to avoid numerical issues.
        Dtype maxval = conf_data[start_idx];
        for (int c = 1; c < num_classes; ++c) {
          maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
        }
        Dtype sum = 0.;
        for (int c = 0; c < num_classes; ++c) {
          sum += std::exp(conf_data[start_idx + c] - maxval);
        }
        Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
        loss = -log(std::max(prob, Dtype(FLT_MIN)));
      } else if (loss_type == MultiBoxLossParameter_ConfLossType_LOGISTIC) {
        int target = 0;
        for (int c = 0; c < num_classes; ++c) {
          if (c == label) {
            target = 1;
          } else {
            target = 0;
          }
          Dtype input = conf_data[start_idx + c];
          loss -= input * (target - (input >= 0)) -
              log(1 + exp(input - 2 * input * (input >= 0)));
        }
      } else {
        LOG(FATAL) << "Unknown conf loss type.";
      }
      conf_loss.push_back(loss);
    }
    conf_data += num_preds_per_class * num_classes;
    all_conf_loss->push_back(conf_loss);
  }
}

// Explicit initialization.
template void ComputeConfLoss(const float* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);
template void ComputeConfLoss(const double* conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);

template <typename Dtype>
void ComputePboxConfLoss(const Dtype* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	const int background_label_id, const PConfLossType loss_type,
	const vector<map<int, vector<int> > >& all_match_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	vector<vector<float> >* all_conf_loss) {
	CHECK_LT(background_label_id, num_classes);
	// CHECK_EQ(num, all_match_indices.size());
	all_conf_loss->clear();
	for (int i = 0; i < num; ++i) {
		vector<float> conf_loss;
		const map<int, vector<int> >& match_indices = all_match_indices[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_classes;
			// Get the label index.
			int label = background_label_id;
			for (map<int, vector<int> >::const_iterator it =
				match_indices.begin(); it != match_indices.end(); ++it) {
				const vector<int>& match_index = it->second;
				CHECK_EQ(match_index.size(), num_preds_per_class);
				if (match_index[p] > -1) {
					CHECK(all_gt_pboxes.find(i) != all_gt_pboxes.end());
					const vector<NormalizedPBox>& gt_pboxes =
						all_gt_pboxes.find(i)->second;
					CHECK_LT(match_index[p], gt_pboxes.size());
					label = gt_pboxes[match_index[p]].label();
					CHECK_GE(label, 0);
					CHECK_NE(label, background_label_id);
					CHECK_LT(label, num_classes);
					// A prior can only be matched to one gt bbox.
					break;
				}
			}
			Dtype loss = 0;
			if (loss_type == MultiPBoxLossParameter_ConfLossType_SOFTMAX) {
				CHECK_GE(label, 0);
				CHECK_LT(label, num_classes);
				// Compute softmax probability.
				// We need to subtract the max to avoid numerical issues.
				Dtype maxval = conf_data[start_idx];
				for (int c = 1; c < num_classes; ++c) {
					maxval = std::max<Dtype>(conf_data[start_idx + c], maxval);
				}
				Dtype sum = 0.;
				for (int c = 0; c < num_classes; ++c) {
					sum += std::exp(conf_data[start_idx + c] - maxval);
				}
				Dtype prob = std::exp(conf_data[start_idx + label] - maxval) / sum;
				loss = -log(std::max(prob, Dtype(FLT_MIN)));
			}
			else if (loss_type == MultiPBoxLossParameter_ConfLossType_LOGISTIC) {
				int target = 0;
				for (int c = 0; c < num_classes; ++c) {
					if (c == label) {
						target = 1;
					}
					else {
						target = 0;
					}
					Dtype input = conf_data[start_idx + c];
					loss -= input * (target - (input >= 0)) -
						log(1 + exp(input - 2 * input * (input >= 0)));
				}
			}
			else {
				LOG(FATAL) << "Unknown conf loss type.";
			}
			conf_loss.push_back(loss);
		}
		conf_data += num_preds_per_class * num_classes;
		all_conf_loss->push_back(conf_loss);
	}
}

// Explicit initialization.
template void ComputePboxConfLoss(const float* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	const int background_label_id, const PConfLossType loss_type,
	const vector<map<int, vector<int> > >& all_match_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	vector<vector<float> >* all_conf_loss);
template void ComputePboxConfLoss(const double* conf_data, const int num,
	const int num_preds_per_class, const int num_classes,
	const int background_label_id, const PConfLossType loss_type,
	const vector<map<int, vector<int> > >& all_match_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	vector<vector<float> >* all_conf_loss);

template <typename Dtype>
void EncodeConfPrediction(const Dtype* conf_data, const int num,
	const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<vector<int> >& all_neg_indices,
	const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
	Dtype* conf_pred_data, Dtype* conf_gt_data) {
	// CHECK_EQ(num, all_match_indices.size());
	// CHECK_EQ(num, all_neg_indices.size());
	// Retrieve parameters.
	CHECK(multibox_loss_param.has_num_classes()) << "Must provide num_classes.";
	const int num_classes = multibox_loss_param.num_classes();
	CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
	const int background_label_id = multibox_loss_param.background_label_id();
	const bool map_object_to_agnostic =
		multibox_loss_param.map_object_to_agnostic();
	if (map_object_to_agnostic) {
		if (background_label_id >= 0) {
			CHECK_EQ(num_classes, 2);
		}
		else {
			CHECK_EQ(num_classes, 1);
		}
	}
	const MiningType mining_type = multibox_loss_param.mining_type();
	bool do_neg_mining;
	if (multibox_loss_param.has_do_neg_mining()) {
		LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
		do_neg_mining = multibox_loss_param.do_neg_mining();
		CHECK_EQ(do_neg_mining,
			mining_type != MultiBoxLossParameter_MiningType_NONE);
	}
	do_neg_mining = mining_type != MultiBoxLossParameter_MiningType_NONE;
	const ConfLossType conf_loss_type = multibox_loss_param.conf_loss_type();
	int count = 0;
	for (int i = 0; i < num; ++i) {
		if (all_gt_bboxes.find(i) != all_gt_bboxes.end()) {
			// Save matched (positive) bboxes scores and labels.
			const map<int, vector<int> >& match_indices = all_match_indices[i];
			for (map<int, vector<int> >::const_iterator it =
				match_indices.begin(); it != match_indices.end(); ++it) {
				const vector<int>& match_index = it->second;
				CHECK_EQ(match_index.size(), num_priors);
				for (int j = 0; j < num_priors; ++j) {
					if (match_index[j] <= -1) {
						continue;
					}
					const int gt_label = map_object_to_agnostic ?
						background_label_id + 1 :
						all_gt_bboxes.find(i)->second[match_index[j]].label();
					int idx = do_neg_mining ? count : j;
					switch (conf_loss_type) {
					case MultiBoxLossParameter_ConfLossType_SOFTMAX:
						conf_gt_data[idx] = gt_label;
						break;
					case MultiBoxLossParameter_ConfLossType_LOGISTIC:
						conf_gt_data[idx * num_classes + gt_label] = 1;
						break;
					default:
						LOG(FATAL) << "Unknown conf loss type.";
					}
					if (do_neg_mining) {
						// Copy scores for matched bboxes.
						caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
							conf_pred_data + count * num_classes);
						++count;
					}
				}
			}
			// Go to next image.
			if (do_neg_mining) {
				// Save negative bboxes scores and labels.
				for (int n = 0; n < all_neg_indices[i].size(); ++n) {
					int j = all_neg_indices[i][n];
					CHECK_LT(j, num_priors);
					caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
						conf_pred_data + count * num_classes);
					switch (conf_loss_type) {
					case MultiBoxLossParameter_ConfLossType_SOFTMAX:
						conf_gt_data[count] = background_label_id;
						break;
					case MultiBoxLossParameter_ConfLossType_LOGISTIC:
						if (background_label_id >= 0 &&
							background_label_id < num_classes) {
							conf_gt_data[count * num_classes + background_label_id] = 1;
						}
						break;
					default:
						LOG(FATAL) << "Unknown conf loss type.";
					}
					++count;
				}
			}
		}
		if (do_neg_mining) {
			conf_data += num_priors * num_classes;
		}
		else {
			conf_gt_data += num_priors;
		}
	}
}
// Explicite initialization.
template void EncodeConfPrediction(const float* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      float* conf_pred_data, float* conf_gt_data);
template void EncodeConfPrediction(const double* conf_data, const int num,
      const int num_priors, const MultiBoxLossParameter& multibox_loss_param,
      const vector<map<int, vector<int> > >& all_match_indices,
      const vector<vector<int> >& all_neg_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      double* conf_pred_data, double* conf_gt_data);

//*******************************************************************************
template <typename Dtype>
void EncodeConfPrediction(const Dtype* conf_data, const int num,
	const int num_priors, const MultiPBoxLossParameter& multipbox_loss_param,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<vector<int> >& all_neg_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	Dtype* conf_pred_data, Dtype* conf_gt_data) {
	// CHECK_EQ(num, all_match_indices.size());
	// CHECK_EQ(num, all_neg_indices.size());
	// Retrieve parameters.
	CHECK(multipbox_loss_param.has_num_classes()) << "Must provide num_classes.";
	const int num_classes = multipbox_loss_param.num_classes();
	CHECK_GE(num_classes, 1) << "num_classes should not be less than 1.";
	const int background_label_id = multipbox_loss_param.background_label_id();
	const bool map_object_to_agnostic =
		multipbox_loss_param.map_object_to_agnostic();
	if (map_object_to_agnostic) {
		if (background_label_id >= 0) {
			CHECK_EQ(num_classes, 2);
		}
		else {
			CHECK_EQ(num_classes, 1);
		}
	}
	const PMiningType mining_type = multipbox_loss_param.mining_type();
	bool do_neg_mining;
	if (multipbox_loss_param.has_do_neg_mining()) {
		LOG(WARNING) << "do_neg_mining is deprecated, use mining_type instead.";
		do_neg_mining = multipbox_loss_param.do_neg_mining();
		CHECK_EQ(do_neg_mining,
			mining_type != MultiPBoxLossParameter_MiningType_NONE);
	}
	do_neg_mining = mining_type != MultiPBoxLossParameter_MiningType_NONE;
	const PConfLossType conf_loss_type = multipbox_loss_param.conf_loss_type();
	int count = 0;
	for (int i = 0; i < num; ++i) {
		if (all_gt_pboxes.find(i) != all_gt_pboxes.end()) {
			// Save matched (positive) pboxes scores and labels.
			const map<int, vector<int> >& match_indices = all_match_indices[i];
			for (map<int, vector<int> >::const_iterator it =
				match_indices.begin(); it != match_indices.end(); ++it) {
				const vector<int>& match_index = it->second;
				CHECK_EQ(match_index.size(), num_priors);
				for (int j = 0; j < num_priors; ++j) {
					if (match_index[j] <= -1) {
						continue;
					}
					const int gt_label = map_object_to_agnostic ?
						background_label_id + 1 :
						all_gt_pboxes.find(i)->second[match_index[j]].label();
					int idx = do_neg_mining ? count : j;
					switch (conf_loss_type) {
					case MultiPBoxLossParameter_ConfLossType_SOFTMAX:
						conf_gt_data[idx] = gt_label;
						break;
					case MultiPBoxLossParameter_ConfLossType_LOGISTIC:
						conf_gt_data[idx * num_classes + gt_label] = 1;
						break;
					default:
						LOG(FATAL) << "Unknown conf loss type.";
					}
					if (do_neg_mining) {
						// Copy scores for matched pboxes.
						caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
							conf_pred_data + count * num_classes);
						++count;
					}
				}
			}
			// Go to next image.
			if (do_neg_mining) {
				// Save negative pboxes scores and labels.
				for (int n = 0; n < all_neg_indices[i].size(); ++n) {
					int j = all_neg_indices[i][n];
					CHECK_LT(j, num_priors);
					caffe_copy<Dtype>(num_classes, conf_data + j * num_classes,
						conf_pred_data + count * num_classes);
					switch (conf_loss_type) {
					case MultiPBoxLossParameter_ConfLossType_SOFTMAX:
						conf_gt_data[count] = background_label_id;
						break;
					case MultiPBoxLossParameter_ConfLossType_LOGISTIC:
						if (background_label_id >= 0 &&
							background_label_id < num_classes) {
							conf_gt_data[count * num_classes + background_label_id] = 1;
						}
						break;
					default:
						LOG(FATAL) << "Unknown conf loss type.";
					}
					++count;
				}
			}
		}
		if (do_neg_mining) {
			conf_data += num_priors * num_classes;
		}
		else {
			conf_gt_data += num_priors;
		}
	}
}
// Explicite initialization.
template void EncodeConfPrediction(const float* conf_data, const int num,
	const int num_priors, const MultiPBoxLossParameter& multipbox_loss_param,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<vector<int> >& all_neg_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	float* conf_pred_data, float* conf_gt_data);
template void EncodeConfPrediction(const double* conf_data, const int num,
	const int num_priors, const MultiPBoxLossParameter& multipbox_loss_param,
	const vector<map<int, vector<int> > >& all_match_indices,
	const vector<vector<int> >& all_neg_indices,
	const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
	double* conf_pred_data, double* conf_gt_data);


template <typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances) {
  prior_bboxes->clear();
  prior_variances->clear();
  for (int i = 0; i < num_priors; ++i) {
    int start_idx = i * 4;
    NormalizedBBox bbox;
    bbox.set_xmin(prior_data[start_idx]);
    bbox.set_ymin(prior_data[start_idx + 1]);
    bbox.set_xmax(prior_data[start_idx + 2]);
    bbox.set_ymax(prior_data[start_idx + 3]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    prior_bboxes->push_back(bbox);
  }

  for (int i = 0; i < num_priors; ++i) {
    int start_idx = (num_priors + i) * 4;
    vector<float> var;
    for (int j = 0; j < 4; ++j) {
      var.push_back(prior_data[start_idx + j]);
    }
    prior_variances->push_back(var);
  }
}

// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);
template void GetPriorBBoxes(const double* prior_data, const int num_priors,
      vector<NormalizedBBox>* prior_bboxes,
      vector<vector<float> >* prior_variances);


//************************************************************************************
template <typename Dtype>
void GetPriorPBoxes(const Dtype* prior_data, const int num_priors,
	vector<NormalizedPBox>* prior_pboxes,
	vector<vector<float> >* prior_variances) {
	prior_pboxes->clear();
	prior_variances->clear();
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = i * 8;
		NormalizedPBox pbox;
		pbox.set_ltopx(prior_data[start_idx]);
		pbox.set_ltopy(prior_data[start_idx + 1]);
		pbox.set_lbottomx(prior_data[start_idx + 2]);
		pbox.set_lbottomy(prior_data[start_idx + 3]);
		pbox.set_rbottomx(prior_data[start_idx + 4]);
		pbox.set_rbottomy(prior_data[start_idx + 5]);
		pbox.set_rtopx(prior_data[start_idx + 6]);
		pbox.set_rtopy(prior_data[start_idx + 7]);
		float pbox_size = PBoxSize(pbox);
		pbox.set_size(pbox_size);
		prior_pboxes->push_back(pbox);
	}

	for (int i = 0; i < num_priors; ++i) {
		int start_idx = (num_priors + i) * 8;
		vector<float> var;
		for (int j = 0; j < 8; ++j) {
			var.push_back(prior_data[start_idx + j]);
			//LOG(INFO) << prior_data[start_idx + j];
		}
		prior_variances->push_back(var);
	}
}

// Explicit initialization.
template void GetPriorPBoxes(const float* prior_data, const int num_priors,
	vector<NormalizedPBox>* prior_pboxes,
	vector<vector<float> >* prior_variances);
template void GetPriorPBoxes(const double* prior_data, const int num_priors,
	vector<NormalizedPBox>* prior_pboxes,
	vector<vector<float> >* prior_variances);


template <typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections) {
  all_detections->clear();
  for (int i = 0; i < num_det; ++i) {
    int start_idx = i * 7;
    int item_id = det_data[start_idx];
    if (item_id == -1) {
      continue;
    }
    int label = det_data[start_idx + 1];
    CHECK_NE(background_label_id, label)
        << "Found background label in the detection results.";
    NormalizedBBox bbox;
    bbox.set_score(det_data[start_idx + 2]);
    bbox.set_xmin(det_data[start_idx + 3]);
    bbox.set_ymin(det_data[start_idx + 4]);
    bbox.set_xmax(det_data[start_idx + 5]);
    bbox.set_ymax(det_data[start_idx + 6]);
    float bbox_size = BBoxSize(bbox);
    bbox.set_size(bbox_size);
    (*all_detections)[item_id][label].push_back(bbox);
  }
}

// Explicit initialization.
template void GetDetectionResults(const float* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);
template void GetDetectionResults(const double* det_data, const int num_det,
      const int background_label_id,
      map<int, map<int, vector<NormalizedBBox> > >* all_detections);

//*****************************************************************************

template <typename Dtype>
void GetPboxDetectionResults(const Dtype* det_data, const int num_det,
	const int background_label_id,
	map<int, map<int, vector<NormalizedPBox> > >* all_detections) {
	all_detections->clear();
	for (int i = 0; i < num_det; ++i) {
		int start_idx = i * 11;
		int item_id = det_data[start_idx];
		if (item_id == -1) {
			continue;
		}
		int label = det_data[start_idx + 1];
		CHECK_NE(background_label_id, label)
			<< "Found background label in the detection results.";
		NormalizedPBox pbox;
		pbox.set_score(det_data[start_idx + 2]);
		pbox.set_ltopx(det_data[start_idx + 3]);
		pbox.set_ltopy(det_data[start_idx + 4]);
		pbox.set_lbottomx(det_data[start_idx + 5]);
		pbox.set_lbottomy(det_data[start_idx + 6]);
		pbox.set_rbottomx(det_data[start_idx + 7]);
		pbox.set_rbottomy(det_data[start_idx + 8]);
		pbox.set_rtopx(det_data[start_idx + 9]);
		pbox.set_rtopy(det_data[start_idx + 10]);

		/*bbox.set_xmin(det_data[start_idx + 3]);
		bbox.set_ymin(det_data[start_idx + 4]);
		bbox.set_xmax(det_data[start_idx + 5]);
		bbox.set_ymax(det_data[start_idx + 6]);*/
		float pbox_size = PBoxSize(pbox);
		pbox.set_size(pbox_size);
		(*all_detections)[item_id][label].push_back(pbox);
	}
}

// Explicit initialization.
template void GetPboxDetectionResults(const float* det_data, const int num_det,
	const int background_label_id,
	map<int, map<int, vector<NormalizedPBox> > >* all_detections);
template void GetPboxDetectionResults(const double* det_data, const int num_det,
	const int background_label_id,
	map<int, map<int, vector<NormalizedPBox> > >* all_detections);

void GetTopKScoreIndex(const vector<float>& scores, const vector<int>& indices,
      const int top_k, vector<pair<float, int> >* score_index_vec) {
  CHECK_EQ(scores.size(), indices.size());

  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    score_index_vec->push_back(std::make_pair(scores[i], indices[i]));
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
                   SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template <typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
      const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
  // Generate index score pairs.
  for (int i = 0; i < num; ++i) {
    if (scores[i] > threshold) {
      score_index_vec->push_back(std::make_pair(scores[i], i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::sort(score_index_vec->begin(), score_index_vec->end(),
            SortScorePairDescend<int>);

  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < score_index_vec->size()) {
    score_index_vec->resize(top_k);
  }
}

template
void GetMaxScoreIndex(const float* scores, const int num, const float threshold,
      const int top_k, vector<pair<float, int> >* score_index_vec);
template
void GetMaxScoreIndex(const double* scores, const int num,
      const float threshold, const int top_k,
      vector<pair<double, int> >* score_index_vec);

void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, const bool reuse_overlaps,
      map<int, map<int, float> >* overlaps, vector<int>* indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  vector<int> idx(boost::counting_iterator<int>(0),
                  boost::counting_iterator<int>(scores.size()));
  vector<pair<float, int> > score_index_vec;
  GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

  // Do nms.
  indices->clear();
  while (score_index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = score_index_vec.front().second;
    const NormalizedBBox& best_bbox = bboxes[best_idx];
    if (BBoxSize(best_bbox) < 1e-5) {
      // Erase small box.
      score_index_vec.erase(score_index_vec.begin());
      continue;
    }
    indices->push_back(best_idx);
    // Erase the best box.
    score_index_vec.erase(score_index_vec.begin());

    if (top_k > -1 && indices->size() >= top_k) {
      // Stop if finding enough bboxes for nms.
      break;
    }

    // Compute overlap between best_bbox and other remaining bboxes.
    // Remove a bbox if the overlap with best_bbox is larger than nms_threshold.
    for (vector<pair<float, int> >::iterator it = score_index_vec.begin();
         it != score_index_vec.end(); ) {
      int cur_idx = it->second;
      const NormalizedBBox& cur_bbox = bboxes[cur_idx];
      if (BBoxSize(cur_bbox) < 1e-5) {
        // Erase small box.
        it = score_index_vec.erase(it);
        continue;
      }
      float cur_overlap = 0.;
      if (reuse_overlaps) {
        if (overlaps->find(best_idx) != overlaps->end() &&
            overlaps->find(best_idx)->second.find(cur_idx) !=
            (*overlaps)[best_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[best_idx][cur_idx];
        } else if (overlaps->find(cur_idx) != overlaps->end() &&
                   overlaps->find(cur_idx)->second.find(best_idx) !=
                   (*overlaps)[cur_idx].end()) {
          // Use the computed overlap.
          cur_overlap = (*overlaps)[cur_idx][best_idx];
        } else {
          cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
          // Store the overlap for future use.
          (*overlaps)[best_idx][cur_idx] = cur_overlap;
        }
      } else {
        cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
      }

      // Remove it if necessary
      if (cur_overlap > threshold) {
        it = score_index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
      const float threshold, const int top_k, vector<int>* indices) {
  bool reuse_overlap = false;
  map<int, map<int, float> > overlaps;
  ApplyNMS(bboxes, scores, threshold, top_k, reuse_overlap, &overlaps, indices);
}

void ApplyNMS(const bool* overlapped, const int num, vector<int>* indices) {
  vector<int> index_vec(boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(num));
  // Do nms.
  indices->clear();
  while (index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = index_vec.front();
    indices->push_back(best_idx);
    // Erase the best box.
    index_vec.erase(index_vec.begin());

    for (vector<int>::iterator it = index_vec.begin(); it != index_vec.end();) {
      int cur_idx = *it;

      // Remove it if necessary
      if (overlapped[best_idx * num + cur_idx]) {
        it = index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void ApplyPboxNMS(const vector<NormalizedPBox>& pboxes, const vector<float>& scores,
	const float threshold, const int top_k, const bool reuse_overlaps,
	map<int, map<int, float> >* overlaps, vector<int>* indices) {
	// Sanity check.
	CHECK_EQ(pboxes.size(), scores.size())
		<< "pboxes and scores have different size.";

	// Get top_k scores (with corresponding indices).
	vector<int> idx(boost::counting_iterator<int>(0),
		boost::counting_iterator<int>(scores.size()));
	vector<pair<float, int> > score_index_vec;
	GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

	// Do nms.
	indices->clear();
	while (score_index_vec.size() != 0) {
		// Get the current highest score box.
		int best_idx = score_index_vec.front().second;
		const NormalizedPBox& best_pbox = pboxes[best_idx];
		if (PBoxSize(best_pbox) < 1e-5) {
			// Erase small box.
			score_index_vec.erase(score_index_vec.begin());
			continue;
		}
		indices->push_back(best_idx);
		// Erase the best box.
		score_index_vec.erase(score_index_vec.begin());

		if (top_k > -1 && indices->size() >= top_k) {
			// Stop if finding enough pboxes for nms.
			break;
		}

		// Compute overlap between best_pbox and other remaining pboxes.
		// Remove a pbox if the overlap with best_pbox is larger than nms_threshold.
		for (vector<pair<float, int> >::iterator it = score_index_vec.begin();
			it != score_index_vec.end(); ) {
			int cur_idx = it->second;
			const NormalizedPBox& cur_pbox = pboxes[cur_idx];
			if (PBoxSize(cur_pbox) < 1e-5) {
				// Erase small box.
				it = score_index_vec.erase(it);
				continue;
			}
			float cur_overlap = 0.;
			if (reuse_overlaps) {
				if (overlaps->find(best_idx) != overlaps->end() &&
					overlaps->find(best_idx)->second.find(cur_idx) !=
					(*overlaps)[best_idx].end()) {
					// Use the computed overlap.
					cur_overlap = (*overlaps)[best_idx][cur_idx];
				}
				else if (overlaps->find(cur_idx) != overlaps->end() &&
					overlaps->find(cur_idx)->second.find(best_idx) !=
					(*overlaps)[cur_idx].end()) {
					// Use the computed overlap.
					cur_overlap = (*overlaps)[cur_idx][best_idx];
				}
				else {
					cur_overlap = JaccardOverlapPbox(best_pbox, cur_pbox);
					// Store the overlap for future use.
					(*overlaps)[best_idx][cur_idx] = cur_overlap;
				}
			}
			else {
				cur_overlap = JaccardOverlapPbox(best_pbox, cur_pbox);
			}

			// Remove it if necessary
			if (cur_overlap > threshold) {
				it = score_index_vec.erase(it);
			}
			else {
				++it;
			}
		}
	}
}

void ApplyPboxNMS(const vector<NormalizedPBox>& pboxes, const vector<float>& scores,
	const float threshold, const int top_k, vector<int>* indices) {
	bool reuse_overlap = false;
	map<int, map<int, float> > overlaps;
	ApplyPboxNMS(pboxes, scores, threshold, top_k, reuse_overlap, &overlaps, indices);
}

void ApplyPboxNMS(const bool* overlapped, const int num, vector<int>* indices) {
  vector<int> index_vec(boost::counting_iterator<int>(0),
                        boost::counting_iterator<int>(num));
  // Do nms.
  indices->clear();
  while (index_vec.size() != 0) {
    // Get the current highest score box.
    int best_idx = index_vec.front();
    indices->push_back(best_idx);
    // Erase the best box.
    index_vec.erase(index_vec.begin());

    for (vector<int>::iterator it = index_vec.begin(); it != index_vec.end();) {
      int cur_idx = *it;

      // Remove it if necessary
      if (overlapped[best_idx * num + cur_idx]) {
        it = index_vec.erase(it);
      } else {
        ++it;
      }
    }
  }
}


inline int clamp(const int v, const int a, const int b) {
  return v < a ? a : v > b ? b : v;
}

void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
      const vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      vector<int>* indices) {
  // Sanity check.
  CHECK_EQ(bboxes.size(), scores.size())
      << "bboxes and scores have different size.";

  // Get top_k scores (with corresponding indices).
  vector<pair<float, int> > score_index_vec;
  GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

void ApplyNMSFast(const vector<NormalizedPBox>& pboxes,
	const vector<float>& scores, const float score_threshold,
	const float nms_threshold, const float eta, const int top_k,
	vector<int>* indices) {
	// Sanity check.
	CHECK_EQ(pboxes.size(), scores.size())
		<< "pboxes and scores have different size.";

	// Get top_k scores (with corresponding indices).
	vector<pair<float, int> > score_index_vec;
	GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

	// Do nms.
	float adaptive_threshold = nms_threshold;
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlapPbox(pboxes[idx], pboxes[kept_idx]);
				keep = overlap <= adaptive_threshold;
			}
			else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}


template <typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices) {
  // Get top_k scores (with corresponding indices).
  vector<pair<Dtype, int> > score_index_vec;
  GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

  // Do nms.
  float adaptive_threshold = nms_threshold;
  indices->clear();
  while (score_index_vec.size() != 0) {
    const int idx = score_index_vec.front().second;
    bool keep = true;
    for (int k = 0; k < indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*indices)[k];
        float overlap = JaccardOverlap(bboxes + idx * 4, bboxes + kept_idx * 4);
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      indices->push_back(idx);
    }
    score_index_vec.erase(score_index_vec.begin());
    if (keep && eta < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= eta;
    }
  }
}

template
void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);
template
void ApplyNMSFast(const double* bboxes, const double* scores, const int num,
      const float score_threshold, const float nms_threshold,
      const float eta, const int top_k, vector<int>* indices);

//***************************************************************************
template <typename Dtype>
void ApplyPboxNMSFast(const Dtype* pboxes, const Dtype* scores, const int num,
	const float score_threshold, const float nms_threshold,
	const float eta, const int top_k, vector<int>* indices) {
	// Get top_k scores (with corresponding indices).
	vector<pair<Dtype, int> > score_index_vec;
	GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

	// Do nms.
	float adaptive_threshold = nms_threshold;
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlapPbox(pboxes + idx * 8, pboxes + kept_idx * 8);
				keep = overlap <= adaptive_threshold;
			}
			else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}

template
void ApplyPboxNMSFast(const float* pboxes, const float* scores, const int num,
	const float score_threshold, const float nms_threshold,
	const float eta, const int top_k, vector<int>* indices);
template
void ApplyPboxNMSFast(const double* pboxes, const double* scores, const int num,
	const float score_threshold, const float nms_threshold,
	const float eta, const int top_k, vector<int>* indices);

void CumSum(const vector<pair<float, int> >& pairs, vector<int>* cumsum) {
  // Sort the pairs based on first item of the pair.
  vector<pair<float, int> > sort_pairs = pairs;
  std::stable_sort(sort_pairs.begin(), sort_pairs.end(),
                   SortScorePairDescend<int>);

  cumsum->clear();
  for (int i = 0; i < sort_pairs.size(); ++i) {
    if (i == 0) {
      cumsum->push_back(sort_pairs[i].second);
    } else {
      cumsum->push_back(cumsum->back() + sort_pairs[i].second);
    }
  }
}

void ComputeAP(const vector<pair<float, int> >& tp, const int num_pos,
               const vector<pair<float, int> >& fp, const string ap_version,
               vector<float>* prec, vector<float>* rec, float* ap) {
  const float eps = 1e-6;
  CHECK_EQ(tp.size(), fp.size()) << "tp must have same size as fp.";
  const int num = tp.size();
  // Make sure that tp and fp have complement value.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(fabs(tp[i].first - fp[i].first), eps);
    CHECK_EQ(tp[i].second, 1 - fp[i].second);
  }
  prec->clear();
  rec->clear();
  *ap = 0;
  if (tp.size() == 0 || num_pos == 0) {
    return;
  }

  // Compute cumsum of tp.
  vector<int> tp_cumsum;
  CumSum(tp, &tp_cumsum);
  CHECK_EQ(tp_cumsum.size(), num);

  // Compute cumsum of fp.
  vector<int> fp_cumsum;
  CumSum(fp, &fp_cumsum);
  CHECK_EQ(fp_cumsum.size(), num);

  // Compute precision.
  for (int i = 0; i < num; ++i) {
    prec->push_back(static_cast<float>(tp_cumsum[i]) /
                    (tp_cumsum[i] + fp_cumsum[i]));
  }

  // Compute recall.
  for (int i = 0; i < num; ++i) {
    CHECK_LE(tp_cumsum[i], num_pos);
    rec->push_back(static_cast<float>(tp_cumsum[i]) / num_pos);
  }

  if (ap_version == "11point") {
    // VOC2007 style for computing AP.
    vector<float> max_precs(11, 0.);
    int start_idx = num - 1;
    for (int j = 10; j >= 0; --j) {
      for (int i = start_idx; i >= 0 ; --i) {
        if ((*rec)[i] < j / 10.) {
          start_idx = i;
          if (j > 0) {
            max_precs[j-1] = max_precs[j];
          }
          break;
        } else {
          if (max_precs[j] < (*prec)[i]) {
            max_precs[j] = (*prec)[i];
          }
        }
      }
    }
    for (int j = 10; j >= 0; --j) {
      *ap += max_precs[j] / 11;
    }
  } else if (ap_version == "MaxIntegral") {
    // VOC2012 or ILSVRC style for computing AP.
    float cur_rec = rec->back();
    float cur_prec = prec->back();
    for (int i = num - 2; i >= 0; --i) {
      cur_prec = std::max<float>((*prec)[i], cur_prec);
      if (fabs(cur_rec - (*rec)[i]) > eps) {
        *ap += cur_prec * fabs(cur_rec - (*rec)[i]);
      }
      cur_rec = (*rec)[i];
    }
    *ap += cur_rec * cur_prec;
  } else if (ap_version == "Integral") {
    // Natural integral.
    float prev_rec = 0.;
    for (int i = 0; i < num; ++i) {
      if (fabs((*rec)[i] - prev_rec) > eps) {
        *ap += (*prec)[i] * fabs((*rec)[i] - prev_rec);
      }
      prev_rec = (*rec)[i];
    }
  } else {
    LOG(FATAL) << "Unknown ap_version: " << ap_version;
  }
}

#ifdef USE_OPENCV
cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f*s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
      r = v; g = p; b = q;
      break;
    default:
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}

// http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically
vector<cv::Scalar> GetColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate,
                              1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

static clock_t start_clock = clock();
static cv::VideoWriter cap_out;

template <typename Dtype>
void VisualizeBBox(const vector<cv::Mat>& images, const Blob<Dtype>* detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file) {
  // Retrieve detections.
  CHECK_EQ(detections->width(), 7);
  const int num_det = detections->height();
  const int num_img = images.size();
  if (num_det == 0 || num_img == 0) {
    return;
  }
  // Comute FPS.
  float fps = num_img / (static_cast<double>(clock() - start_clock) /
          CLOCKS_PER_SEC);

  const Dtype* detections_data = detections->cpu_data();
  const int width = images[0].cols;
  const int height = images[0].rows;
  vector<LabelBBox> all_detections(num_img);
  for (int i = 0; i < num_det; ++i) {
    const int img_idx = detections_data[i * 7];
    CHECK_LT(img_idx, num_img);
    const int label = detections_data[i * 7 + 1];
    const float score = detections_data[i * 7 + 2];
    if (score < threshold) {
      continue;
    }
    NormalizedBBox bbox;
    bbox.set_xmin(detections_data[i * 7 + 3] * width);
    bbox.set_ymin(detections_data[i * 7 + 4] * height);
    bbox.set_xmax(detections_data[i * 7 + 5] * width);
    bbox.set_ymax(detections_data[i * 7 + 6] * height);
    bbox.set_score(score);
    all_detections[img_idx][label].push_back(bbox);
  }

  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];
  for (int i = 0; i < num_img; ++i) {
    cv::Mat image = images[i];
    // Show FPS.
    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
    cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                    &baseline);
    cv::rectangle(image, cv::Point(0, 0),
                  cv::Point(text.width, text.height + baseline),
                  CV_RGB(255, 255, 255), CV_FILLED);
    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    for (map<int, vector<NormalizedBBox> >::iterator it =
         all_detections[i].begin(); it != all_detections[i].end(); ++it) {
      int label = it->first;
      string label_name = "Unknown";
      if (label_to_display_name.find(label) != label_to_display_name.end()) {
        label_name = label_to_display_name.find(label)->second;
      }
      CHECK_LT(label, colors.size());
      const cv::Scalar& color = colors[label];
      const vector<NormalizedBBox>& bboxes = it->second;
      for (int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin(), bboxes[j].ymin());
        cv::Point bottom_right_pt(bboxes[j].xmax(), bboxes[j].ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score());
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                        &baseline);
        cv::rectangle(
            image, bottom_left_pt + cv::Point(0, 0),
            bottom_left_pt + cv::Point(text.width, -text.height-baseline),
            color, CV_FILLED);
        cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
      }
    }
    // Save result if required.
    if (!save_file.empty()) {
      if (!cap_out.isOpened()) {
        cv::Size size(image.size().width, image.size().height);
        cv::VideoWriter outputVideo(save_file, CV_FOURCC('D', 'I', 'V', 'X'),
            30, size, true);
        cap_out = outputVideo;
      }
      cap_out.write(image);
    }
    cv::imshow("detections", image);
    if (cv::waitKey(1) == 27) {
      raise(SIGINT);
    }
  }
  start_clock = clock();
}

template
void VisualizeBBox(const vector<cv::Mat>& images,
                   const Blob<float>* detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file);
template
void VisualizeBBox(const vector<cv::Mat>& images,
                   const Blob<double>* detections,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file);


template <typename Dtype>
void VisualizePBox(const vector<cv::Mat>& images, const Blob<Dtype>* detections,
	const float threshold, const vector<cv::Scalar>& colors,
	const map<int, string>& label_to_display_name,
	const string& save_file) {
	// Retrieve detections.
	CHECK_EQ(detections->width(), 11);
	const int num_det = detections->height();
	const int num_img = images.size();
	if (num_det == 0 || num_img == 0) {
		return;
	}
	// Comute FPS.
	float fps = num_img / (static_cast<double>(clock() - start_clock) /
		CLOCKS_PER_SEC);

	const Dtype* detections_data = detections->cpu_data();
	const int width = images[0].cols;
	const int height = images[0].rows;
	vector<LabelPBox> all_detections(num_img);
	for (int i = 0; i < num_det; ++i) {
		const int img_idx = detections_data[i * 11];
		CHECK_LT(img_idx, num_img);
		const int label = detections_data[i * 11 + 1];
		const float score = detections_data[i * 11 + 2];
		if (score < threshold) {
			continue;
		}
		NormalizedPBox pbox;
		pbox.set_ltopx(detections_data[i * 11 + 3] * width);
		pbox.set_ltopy(detections_data[i * 11 + 4] * height);
		pbox.set_lbottomx(detections_data[i * 11 + 5] * width);
		pbox.set_lbottomy(detections_data[i * 11 + 6] * height);
		pbox.set_rbottomx(detections_data[i * 11 + 7] * width);
		pbox.set_rbottomy(detections_data[i * 11 + 8] * height);
		pbox.set_rtopx(detections_data[i * 11 + 9] * width);
		pbox.set_rtopy(detections_data[i * 11 + 10] * height);
		/*
		bbox.set_xmin(detections_data[i * 7 + 3] * width);
		bbox.set_ymin(detections_data[i * 7 + 4] * height);
		bbox.set_xmax(detections_data[i * 7 + 5] * width);
		bbox.set_ymax(detections_data[i * 7 + 6] * height);*/
		pbox.set_score(score);
		all_detections[img_idx][label].push_back(pbox);
	}

	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	double scale = 1;
	int thickness = 2;
	int baseline = 0;
	char buffer[50];
	for (int i = 0; i < num_img; ++i) {
		cv::Mat image = images[i];
		// Show FPS.
		snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
		cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
			&baseline);
		cv::rectangle(image, cv::Point(0, 0),
			cv::Point(text.width, text.height + baseline),
			CV_RGB(255, 255, 255), CV_FILLED);
		cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
			fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
		// Draw pboxes.
		for (map<int, vector<NormalizedPBox> >::iterator it =
			all_detections[i].begin(); it != all_detections[i].end(); ++it) {
			int label = it->first;
			string label_name = "Unknown";
			if (label_to_display_name.find(label) != label_to_display_name.end()) {
				label_name = label_to_display_name.find(label)->second;
			}
			CHECK_LT(label, colors.size());
			const cv::Scalar& color = colors[label];
			const vector<NormalizedPBox>& pboxes = it->second;
			for (int j = 0; j < pboxes.size(); ++j) {
				cv::Point top_left_pt(pboxes[j].ltopx(), pboxes[j].ltopy());
				cv::Point bottom_left_pt(pboxes[j].lbottomx(), pboxes[j].lbottomy());
				cv::Point bottom_right_pt(pboxes[j].rbottomx(), pboxes[j].rbottomy());
				cv::Point top_right_pt(pboxes[j].rtopx(), pboxes[j].rtopy());
				cv::line(image, top_left_pt, bottom_left_pt, color, 4);
				cv::line(image, bottom_left_pt, bottom_right_pt, color, 4);
				cv::line(image, bottom_right_pt, top_right_pt, color, 4);
				cv::line(image, top_right_pt, top_left_pt, color, 4);
				//cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
				//cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
				snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
					pboxes[j].score());
				cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
					&baseline);
				cv::rectangle(
					image, bottom_left_pt + cv::Point(0, 0),
					bottom_left_pt + cv::Point(text.width, -text.height - baseline),
					color, CV_FILLED);
				cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
					fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
			}
		}
		// Save result if required.
		if (!save_file.empty()) {
			if (!cap_out.isOpened()) {
				cv::Size size(image.size().width, image.size().height);
				cv::VideoWriter outputVideo(save_file, CV_FOURCC('D', 'I', 'V', 'X'),
					30, size, true);
				cap_out = outputVideo;
			}
			cap_out.write(image);
		}
		cv::imshow("detections", image);
		if (cv::waitKey(1) == 27) {
			raise(SIGINT);
		}
	}
	start_clock = clock();
}

template
void VisualizePBox(const vector<cv::Mat>& images,
const Blob<float>* detections,
const float threshold, const vector<cv::Scalar>& colors,
const map<int, string>& label_to_display_name,
const string& save_file);
template
void VisualizePBox(const vector<cv::Mat>& images,
const Blob<double>* detections,
const float threshold, const vector<cv::Scalar>& colors,
const map<int, string>& label_to_display_name,
const string& save_file);


#endif  // USE_OPENCV

}  // namespace caffe