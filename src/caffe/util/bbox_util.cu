#include <algorithm>
#include <functional>
#include <map>
#include <vector>

#include "thrust/functional.h"
#include "thrust/sort.h"

#include "caffe/common.hpp"
#include "caffe/util/bbox_util.hpp"

namespace caffe {

template <typename Dtype>
__host__ __device__ Dtype BBoxSizeGPU(const Dtype* bbox,
    const bool normalized) {
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

template __host__ __device__ float BBoxSizeGPU(const float* bbox,
    const bool normalized);
template __host__ __device__ double BBoxSizeGPU(const double* bbox,
    const bool normalized);
//**********************************************************************
template <typename Dtype>
__host__ __device__ Dtype PBoxSizeGPU(const Dtype* pbox,
    const bool normalized) {
	
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

template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapGPU(const Dtype* bbox1,
    const Dtype* bbox2) {
  if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else {
    const Dtype inter_xmin = max(bbox1[0], bbox2[0]);
    const Dtype inter_ymin = max(bbox1[1], bbox2[1]);
    const Dtype inter_xmax = min(bbox1[2], bbox2[2]);
    const Dtype inter_ymax = min(bbox1[3], bbox2[3]);

    const Dtype inter_width = inter_xmax - inter_xmin;
    const Dtype inter_height = inter_ymax - inter_ymin;
    const Dtype inter_size = inter_width * inter_height;

    const Dtype bbox1_size = BBoxSizeGPU(bbox1);
    const Dtype bbox2_size = BBoxSizeGPU(bbox2);

    return inter_size / (bbox1_size + bbox2_size - inter_size);
  }
}

template __host__ __device__ float JaccardOverlapGPU(const float* bbox1,
    const float* bbox2);
template __host__ __device__ double JaccardOverlapGPU(const double* bbox1,
    const double* bbox2);

//****************************************************************	
template <typename Dtype>
__host__ __device__ Dtype JaccardOverlapPboxGPU(const Dtype* pbox1,
    const Dtype* pbox2) {
  /* if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
      bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
    return Dtype(0.);
  } else { */
	  
  if(PBoxSizeGPU<Dtype>(pbox1)==0.f || PBoxSizeGPU<Dtype>(pbox2)==0.f) return 0.f;
	 // float d1, d2, d3, d4;
	const Dtype d1 = sqrtf((powf((pbox1[0] - pbox2[0]), 2.f) + powf((pbox1[1] - pbox2[1]), 2.f)));
	const Dtype d2 = sqrtf((powf((pbox1[2] - pbox2[2]), 2.f) + powf((pbox1[3] - pbox2[3]), 2.f)));
	const Dtype d3 = sqrtf((powf((pbox1[4] - pbox2[4]), 2.f) + powf((pbox1[5] - pbox2[5]), 2.f)));
	const Dtype d4 = sqrtf((powf((pbox1[6] - pbox2[6]), 2.f) + powf((pbox1[7] - pbox2[7]), 2.f)));

	const Dtype d = max(max(max(d1, d2), d3), d4);

	// const Dtype D1 = sqrtf((powf((pbox1[0] - pbox1[2]), 2.f) + powf((pbox1[1] - pbox1[3]), 2.f)));
	// const Dtype D2 = sqrtf((powf((pbox1[2] - pbox1[4]), 2.f) + powf((pbox1[3] - pbox1[5]), 2.f)));
	// const Dtype D3 = sqrtf((powf((pbox1[4] - pbox1[6]), 2.f) + powf((pbox1[5] - pbox1[7]), 2.f)));
	// const Dtype D4 = sqrtf((powf((pbox1[6] - pbox1[0]), 2.f) + powf((pbox1[7] - pbox1[1]), 2.f)));

	// const Dtype D = min(min(min(D1, D2), D3), D4);

  //return 1.f - expf(-D / d);
  if(d>1.f) return 0.f;
  return 1.f - d;
  //}
}

template __host__ __device__ float JaccardOverlapPboxGPU(const float* pbox1,
    const float* pbox2);
template __host__ __device__ double JaccardOverlapPboxGPU(const double* pbox1,
    const double* pbox2);	

template <typename Dtype>
__device__ Dtype Min(const Dtype x, const Dtype y) {
  return x < y ? x : y;
}

template <typename Dtype>
__device__ Dtype Max(const Dtype x, const Dtype y) {
  return x > y ? x : y;
}

template <typename Dtype>
__device__ Dtype P2PDistanceGPU(const Dtype p1x, const Dtype p1y, const Dtype p2x, const Dtype p2y)
{
	return sqrtf(powf((p1x-p2x), 2.f) + powf((p1y-p2y), 2.f));
}
/*template <typename Dtype>*/
/*__device__ Dtype Pow(const Dtype x, const Dtype n) {*/
    /*Dtype result = Dtype(1.);*/
    /*for (int i = 0;i<n; i++) {*/
        /*result = x*result;*/
    /*}*/
    /*return result;*/
/*}*/

template <typename Dtype>
__device__ void ClipBBoxGPU(const Dtype* bbox, Dtype* clip_bbox) {
  for (int i = 0; i < 4; ++i) {
    clip_bbox[i] = Max(Min(bbox[i], Dtype(1.)), Dtype(0.));
  }
}

template __device__ void ClipBBoxGPU(const float* bbox, float* clip_bbox);
template __device__ void ClipBBoxGPU(const double* bbox, double* clip_bbox);

//***************************************************************************
template <typename Dtype>
__device__ void ClipPBoxGPU(const Dtype* pbox, Dtype* clip_pbox) {
  for (int i = 0; i < 8; ++i) {
    clip_pbox[i] = Max(Min(pbox[i], Dtype(1.)), Dtype(0.));
  }
}

template __device__ void ClipPBoxGPU(const float* pbox, float* clip_pbox);
template __device__ void ClipPBoxGPU(const double* pbox, double* clip_pbox);


template <typename Dtype>
__global__ void DecodeBBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 4;
    const int c = (index / 4) % num_loc_classes;
    const int d = (index / 4 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 4;
    const int vi = pi + num_priors * 4;
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype xmin = loc_data[index - i];
      const Dtype ymin = loc_data[index - i + 1];
      const Dtype xmax = loc_data[index - i + 2];
      const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_bbox_center_x, decode_bbox_center_y;
      Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_bbox_center_x = xmin * prior_width + prior_center_x;
        decode_bbox_center_y = ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(xmax) * prior_width;
        decode_bbox_height = exp(ymax) * prior_height;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_bbox_center_x =
          prior_data[vi] * xmin * prior_width + prior_center_x;
        decode_bbox_center_y =
          prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        decode_bbox_width =
          exp(prior_data[vi + 2] * xmax) * prior_width;
        decode_bbox_height =
          exp(prior_data[vi + 3] * ymax) * prior_height;
      }

      switch (i) {
        case 0:
          bbox_data[index] = decode_bbox_center_x - decode_bbox_width / 2.;
          break;
        case 1:
          bbox_data[index] = decode_bbox_center_y - decode_bbox_height / 2.;
          break;
        case 2:
          bbox_data[index] = decode_bbox_center_x + decode_bbox_width / 2.;
          break;
        case 3:
          bbox_data[index] = decode_bbox_center_y + decode_bbox_height / 2.;
          break;
      }
    } else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    } else {
      // Unknown code type.
    }
    if (clip_bbox) {
      bbox_data[index] = max(min(bbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
void DecodeBBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, Dtype* bbox_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodeBBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data, code_type,
      variance_encoded_in_target, num_priors, share_location, num_loc_classes,
      background_label_id, clip_bbox, bbox_data);
  CUDA_POST_KERNEL_CHECK;
}

template void DecodeBBoxesGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, float* bbox_data);
template void DecodeBBoxesGPU(const int nthreads,
          const double* loc_data, const double* prior_data,
          const CodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_bbox, double* bbox_data);
		  
//*********************************************************************************
template <typename Dtype>
__global__ void DecodePBoxesKernel(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const PCodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_pbox, Dtype* pbox_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % 8;
    const int c = (index / 8) % num_loc_classes;
    const int d = (index / 8 / num_loc_classes) % num_priors;
    if (!share_location && c == background_label_id) {
      // Ignore background class if not share_location.
      return;
    }
    const int pi = d * 8;
    const int vi = pi + num_priors * 8;
    if (code_type == PriorPBoxParameter_CodeType_CORNER) {
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        pbox_data[index] = prior_data[pi + i] + loc_data[index];
      } else {
        // variance is encoded in pbox, we need to scale the offset accordingly.
        pbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i];
      }
    } else if (code_type == PriorPBoxParameter_CodeType_CENTER_SIZE_ANGLE) {
      const Dtype p_ltopx = prior_data[pi];
      const Dtype p_ltopy = prior_data[pi + 1];
      const Dtype p_lbottomx = prior_data[pi + 2];
      const Dtype p_lbottomy = prior_data[pi + 3];
      const Dtype p_rbottomx = prior_data[pi + 4];
      const Dtype p_rbottomy = prior_data[pi + 5];
      const Dtype p_rtopx = prior_data[pi + 6];
      const Dtype p_rtopy = prior_data[pi + 7];
 
    	// const Dtype prior_dlt, prior_dlb, prior_drb, prior_drt;
	  	// const Dtype prior_center_x, prior_center_y;
		  // const Dtype prior_alpha, prior_beta;
			const Dtype ltrbx = p_ltopx - p_rbottomx; //prior_pbox.ltopx() - prior_pbox.rbottomx();
			const Dtype ltrby = p_ltopy - p_rbottomy;//prior_pbox.ltopy() - prior_pbox.rbottomy();
			const Dtype lbrtx = p_rtopx - p_lbottomx;//prior_pbox.rtopx() - prior_pbox.lbottomx();
			const Dtype lbrty = p_rtopy - p_lbottomy;//prior_pbox.rtopy() - prior_pbox.lbottomy();
			//float width = bbox.xmax() - bbox.xmin(); //float height = bbox.ymax() - bbox.ymin(); 
        const Dtype prior_ltrb = sqrtf(powf(ltrbx, 2.) + powf(ltrby, 2.));
				const Dtype prior_lbrt = sqrtf(powf(lbrtx, 2.) + powf(lbrty, 2.));
				const Dtype prior_alpha = atan2f(lbrtx, lbrty) -  atan2f(-ltrbx, -ltrby);
      
      // Dtype * p = NULL;
      // size_t size = 8 * sizeof(Dtype);
      // cudaMalloc((void**)&p, size);
      Dtype p[8];
		// NormalizedPBox p;
		p[0] = (p_ltopx);
		p[1] = (p_ltopy);
		p[2] = (p_lbottomx);
		p[3] = (p_lbottomy);
		p[4] = (p_lbottomx);
		p[5] = (p_lbottomy);
		p[6] = (p_rtopx);
		p[7] = (p_rtopy);
		// const Dtype area1, area2;
		const Dtype area1 = PBoxSizeGPU<Dtype>(p);
		p[0] = (p_rtopx);
		p[1] = (p_rtopy);
	  p[4] = (p_rbottomx);
		p[5] = (p_rbottomy);
		const Dtype area2 = PBoxSizeGPU<Dtype>(p);
		const Dtype k = area2 / area1;
    // cudaFree(p);
    delete p;

		const Dtype prior_center_x = p_ltopx + (p_rbottomx - p_ltopx) /(1+k);
		const Dtype prior_center_y = p_ltopy + (p_rbottomy - p_ltopy) /(1+k);
		// prior_center_y = prior_pbox.ltopy() + (prior_pbox.rbottomy()-prior_pbox.ltopy()) /(1+k);

    // Dtype* prior_points = NULL;
    // size = 10 * sizeof(Dtype);
    // cudaMalloc((void**)& prior_points, size);
    Dtype prior_points[10];

		prior_points[0] = prior_center_x; 
    prior_points[1] = prior_center_y;
    prior_points[2] = p_ltopx; 
    prior_points[3] = p_ltopy;//prior_pbox.ltopx(), prior_pbox.ltopy(),
    prior_points[4] = p_lbottomx;
    prior_points[5] = p_lbottomy; //prior_pbox.lbottomx(), prior_pbox.lbottomy(),
    prior_points[6] = p_rbottomx;
    prior_points[7] = p_rbottomy;//prior_pbox.rbottomx(), prior_pbox.rbottomy(),
    prior_points[8] = p_rtopx;
    prior_points[9] = p_rtopy;//prior_pbox.rtopx(), prior_pbox.rtopy()};

		const Dtype prior_dlt = P2PDistanceGPU<Dtype>(prior_points[0],prior_points[1],prior_points[2],prior_points[3] );
		const Dtype prior_dlb = P2PDistanceGPU<Dtype>(prior_points[0],prior_points[1],prior_points[4],prior_points[5] );
		const Dtype prior_drb = P2PDistanceGPU<Dtype>(prior_points[0],prior_points[1],prior_points[6],prior_points[7] );
		const Dtype prior_drt = P2PDistanceGPU<Dtype>(prior_points[0],prior_points[1],prior_points[8],prior_points[9] );
    // cudaFree(prior_points);
    delete prior_points;
		// CHECK_GT(prior_dlt,0);
		// CHECK_GT(prior_dlb,0);
		// CHECK_GT(prior_drb,0);
		// CHECK_GT(prior_drt,0);

// beta (-pi, pi)
		const Dtype prior_beta = atan2f((p_rtopy- prior_center_y), (p_rtopx - prior_center_x));

		// const Dtype prior_mean_width, prior_mean_height;
		const Dtype prior_mean_width = (fabsf((prior_dlb+prior_drt)*cosf(prior_beta))
												 + fabsf((prior_dlt+prior_drb)*cosf(prior_alpha-prior_beta)))/2.;
		const Dtype prior_mean_height = (fabsf((prior_dlb+prior_drt)*sinf(prior_beta))
												 + fabsf((prior_dlt+prior_drb)*sinf(prior_alpha-prior_beta)))/2.;
		//CHECK_GT(prior_mean_width, 0);
		// CHECK_GT(prior_mean_height, 0);


      // const Dtype p_xmin = prior_data[pi];
      // const Dtype p_ymin = prior_data[pi + 1];
      // const Dtype p_xmax = prior_data[pi + 2];
      // const Dtype p_ymax = prior_data[pi + 3];
      // const Dtype prior_width = p_xmax - p_xmin;
      // const Dtype prior_height = p_ymax - p_ymin;
      // const Dtype prior_center_x = (p_xmin + p_xmax) / 2.;
      // const Dtype prior_center_y = (p_ymin + p_ymax) / 2.;

      const Dtype ltopx = loc_data[index - i];
      const Dtype ltopy = loc_data[index - i + 1];
      const Dtype lbottomx = loc_data[index - i + 2];
      const Dtype lbottomy = loc_data[index - i + 3];
      const Dtype rbottomx = loc_data[index - i + 4];
      const Dtype rbottomy = loc_data[index - i + 5];
      const Dtype rtopx = loc_data[index - i + 6];
      const Dtype rtopy = loc_data[index - i + 7];
      // const Dtype xmin = loc_data[index - i];
      // const Dtype ymin = loc_data[index - i + 1];
      // const Dtype xmax = loc_data[index - i + 2];
      // const Dtype ymax = loc_data[index - i + 3];

      Dtype decode_center_x, decode_center_y;
      Dtype decode_dlt, decode_dlb, decode_drb, decode_drt;
      Dtype decode_alpha, decode_beta;
      // Dtype decode_bbox_width, decode_bbox_height;
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to retore the offset
        // predictions.
        decode_center_x = ltopx * prior_mean_width + prior_center_x;// xmin * prior_width + prior_center_x;
        decode_center_y = ltopy * prior_mean_height + prior_center_y;// ymin * prior_height + prior_center_y;
        decode_dlt = expf(lbottomx) * prior_ltrb;
        decode_dlb = expf(lbottomy) * prior_lbrt;
        decode_drb = expf(rbottomx) * prior_ltrb;
        decode_drt = expf(rbottomy) * prior_lbrt;
        decode_alpha = rtopx + prior_alpha;
        decode_beta = rtopy + prior_beta;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        decode_center_x = prior_data[vi] * ltopx * prior_mean_width + prior_center_x;// xmin * prior_width + prior_center_x;
        decode_center_y = prior_data[vi + 1] * ltopy * prior_mean_height + prior_center_y;// ymin * prior_height + prior_center_y;
        decode_dlt = expf(prior_data[vi + 2] * lbottomx) * prior_ltrb;
        decode_dlb = expf(prior_data[vi + 3] * lbottomy) * prior_lbrt;
        decode_drb = expf(prior_data[vi + 4] * rbottomx) * prior_ltrb;
        decode_drt = expf(prior_data[vi + 5] * rbottomy) * prior_lbrt;
        decode_alpha = prior_data[vi + 6] * rtopx + prior_alpha;
        decode_beta = prior_data[vi + 7] * rtopy + prior_beta;

        // decode_bbox_center_x =
        //   prior_data[vi] * xmin * prior_width + prior_center_x;
        // decode_bbox_center_y =
        //   prior_data[vi + 1] * ymin * prior_height + prior_center_y;
        // decode_bbox_width =
        //   exp(prior_data[vi + 2] * xmax) * prior_width;
        // decode_bbox_height =
        //   exp(prior_data[vi + 3] * ymax) * prior_height;
      }
				// float ltx, lty, lbx, lby, rbx, rby, rtx, rty;
				const Dtype ltx = decode_center_x - decode_dlt * cosf(decode_alpha - decode_beta);
				const Dtype lty = decode_center_y + decode_dlt * sinf(decode_alpha - decode_beta);
				const Dtype lbx = decode_center_x - decode_dlb * cosf(decode_beta);
				const Dtype lby = decode_center_y - decode_dlb * sinf(decode_beta);
				const Dtype rbx = decode_center_x + decode_drb * cosf(decode_alpha - decode_beta);
				const Dtype rby = decode_center_y - decode_drb * sinf(decode_alpha - decode_beta);
				const Dtype rtx = decode_center_x + decode_drt * cosf(decode_beta);
				const Dtype rty = decode_center_y + decode_drt * sinf(decode_beta);


      switch (i) {
        case 0:
          pbox_data[index] = ltx;
          break;
        case 1:
          pbox_data[index] = lty;
          break;
        case 2:
          pbox_data[index] = lbottomx;
          break;
        case 3:
          pbox_data[index] = lbottomy;
          break;
        case 4:
          pbox_data[index] = rbottomx;
          break;
        case 5:
          pbox_data[index] = rbottomy;
          break;
        case 6:
          pbox_data[index] = rtx;
          break;
        case 7:
          pbox_data[index] = rty;
          break;
        
      }
    }/* else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
      const Dtype p_xmin = prior_data[pi];
      const Dtype p_ymin = prior_data[pi + 1];
      const Dtype p_xmax = prior_data[pi + 2];
      const Dtype p_ymax = prior_data[pi + 3];
      const Dtype prior_width = p_xmax - p_xmin;
      const Dtype prior_height = p_ymax - p_ymin;
      Dtype p_size;
      if (i == 0 || i == 2) {
        p_size = prior_width;
      } else {
        p_size = prior_height;
      }
      if (variance_encoded_in_target) {
        // variance is encoded in target, we simply need to add the offset
        // predictions.
        bbox_data[index] = prior_data[pi + i] + loc_data[index] * p_size;
      } else {
        // variance is encoded in bbox, we need to scale the offset accordingly.
        bbox_data[index] =
          prior_data[pi + i] + loc_data[index] * prior_data[vi + i] * p_size;
      }
    }  */
	else {
      // Unknown code type.
    }
    if (clip_pbox) {
      pbox_data[index] = max(min(pbox_data[index], Dtype(1.)), Dtype(0.));
    }
  }
}

template <typename Dtype>
void DecodePBoxesGPU(const int nthreads,
          const Dtype* loc_data, const Dtype* prior_data,
          const PCodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_pbox, Dtype* pbox_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  DecodePBoxesKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, loc_data, prior_data, code_type,
      variance_encoded_in_target, num_priors, share_location, num_loc_classes,
      background_label_id, clip_pbox, pbox_data);
  CUDA_POST_KERNEL_CHECK;
}

template void DecodePBoxesGPU(const int nthreads,
          const float* loc_data, const float* prior_data,
          const PCodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_pbox, float* pbox_data);
template void DecodePBoxesGPU(const int nthreads,
          const double* loc_data, const double* prior_data,
          const PCodeType code_type, const bool variance_encoded_in_target,
          const int num_priors, const bool share_location,
          const int num_loc_classes, const int background_label_id,
          const bool clip_pbox, double* pbox_data);
		  

template <typename Dtype>
__global__ void PermuteDataKernel(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int i = index % num_dim;
    const int c = (index / num_dim) % num_classes;
    const int d = (index / num_dim / num_classes) % num_data;
    const int n = index / num_dim / num_classes / num_data;
    const int new_index = ((n * num_classes + c) * num_data + d) * num_dim + i;
    new_data[new_index] = data[index];
  }
}

template <typename Dtype>
void PermuteDataGPU(const int nthreads,
          const Dtype* data, const int num_classes, const int num_data,
          const int num_dim, Dtype* new_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  PermuteDataKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, data, num_classes, num_data,
      num_dim, new_data);
  CUDA_POST_KERNEL_CHECK;
}

template void PermuteDataGPU(const int nthreads,
          const float* data, const int num_classes, const int num_data,
          const int num_dim, float* new_data);
template void PermuteDataGPU(const int nthreads,
          const double* data, const int num_classes, const int num_data,
          const int num_dim, double* new_data);

template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_data, const Dtype* channel_max,
    Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] = channel_data[index] - channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
void SoftMaxGPU(const Dtype* data, const int outer_num,
    const int channels, const int inner_num, Dtype* prob) {
  vector<int> shape(4, 1);
  shape[0] = outer_num;
  shape[1] = channels;
  shape[2] = inner_num;
  Blob<Dtype> scale(shape);
  Dtype* scale_data = scale.mutable_gpu_data();
  int count = outer_num * channels * inner_num;
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num * inner_num),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num, channels, inner_num, data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num, channels, inner_num,
      data, scale_data, prob);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, prob, prob);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num * inner_num),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num, channels, inner_num, prob,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num, channels, inner_num,
      scale_data, prob);
}

template void SoftMaxGPU(const float* data, const int outer_num,
    const int channels, const int inner_num, float* prob);
template void SoftMaxGPU(const double* data, const int outer_num,
    const int channels, const int inner_num, double* prob);

	
template <typename Dtype>
__global__ void ComputeOverlappedKernel(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_bboxes;
    const int i = (index / num_bboxes) % num_bboxes;
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    const int c = (index / num_bboxes / num_bboxes) % num_classes;
    const int n = index / num_bboxes / num_bboxes / num_classes;
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = ((n * num_bboxes + i) * num_classes + c) * 4;
    const int start_loc_j = ((n * num_bboxes + j) * num_classes + c) * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedGPU(const int nthreads,
          const Dtype* bbox_data, const int num_bboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, num_bboxes, num_classes,
      overlap_threshold, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedGPU(const int nthreads,
          const float* bbox_data, const int num_bboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);
template void ComputeOverlappedGPU(const int nthreads,
          const double* bbox_data, const int num_bboxes, const int num_classes,
          const double overlap_threshold, bool* overlapped_data);

//******************************************************************************
template <typename Dtype>
__global__ void ComputeOverlappedPboxKernel(const int nthreads,
          const Dtype* pbox_data, const int num_pboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_pboxes;
    const int i = (index / num_pboxes) % num_pboxes;
    if (i == j) {
      // Ignore same pbox.
      return;
    }
    const int c = (index / num_pboxes / num_pboxes) % num_classes;
    const int n = index / num_pboxes / num_pboxes / num_classes;
    // Compute overlap between i-th pbox and j-th pbox.
    const int start_loc_i = ((n * num_pboxes + i) * num_classes + c) * 8;
    const int start_loc_j = ((n * num_pboxes + j) * num_classes + c) * 8;
    const Dtype overlap = JaccardOverlapPboxGPU<Dtype>(pbox_data + start_loc_i,
        pbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedPboxGPU(const int nthreads,
          const Dtype* pbox_data, const int num_pboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedPboxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, pbox_data, num_pboxes, num_classes,
      overlap_threshold, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedPboxGPU(const int nthreads,
          const float* pbox_data, const int num_pboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);
template void ComputeOverlappedPboxGPU(const int nthreads,
          const double* pbox_data, const int num_pboxes, const int num_classes,
          const double overlap_threshold, bool* overlapped_data);		  

template <typename Dtype>
__global__ void ComputeOverlappedByIdxKernel(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_idx;
    const int i = (index / num_idx);
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = idx[i] * 4;
    const int start_loc_j = idx[j] * 4;
    const Dtype overlap = JaccardOverlapGPU<Dtype>(bbox_data + start_loc_i,
        bbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedByIdxGPU(const int nthreads,
          const Dtype* bbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedByIdxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bbox_data, overlap_threshold,
      idx, num_idx, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedByIdxGPU(const int nthreads,
          const float* bbox_data, const float overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
template void ComputeOverlappedByIdxGPU(const int nthreads,
          const double* bbox_data, const double overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
		  
template <typename Dtype>
__global__ void ComputeOverlappedPboxByIdxKernel(const int nthreads,
          const Dtype* pbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_idx;
    const int i = (index / num_idx);
    if (i == j) {
      // Ignore same pbox.
      return;
    }
    // Compute overlap between i-th pbox and j-th pbox.
    const int start_loc_i = idx[i] * 8;
    const int start_loc_j = idx[j] * 8;
    const Dtype overlap = JaccardOverlapPboxGPU<Dtype>(pbox_data + start_loc_i,
        pbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const Dtype* pbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedPboxByIdxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, pbox_data, overlap_threshold,
      idx, num_idx, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const float* pbox_data, const float overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
template void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const double* pbox_data, const double overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);

/* *****************************************************************************
template <typename Dtype>
__global__ void ComputeOverlappedPboxKernel(const int nthreads,
          const Dtype* pbox_data, const int num_pboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_pboxes;
    const int i = (index / num_pboxes) % num_pboxes;
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    const int c = (index / num_pboxes / num_pboxes) % num_classes;
    const int n = index / num_pboxes / num_pboxes / num_classes;
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = ((n * num_bboxes + i) * num_classes + c) * 8;
    const int start_loc_j = ((n * num_bboxes + j) * num_classes + c) * 8;
    const Dtype overlap = JaccardOverlapPboxGPU<Dtype>(pbox_data + start_loc_i,
        pbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedPboxGPU(const int nthreads,
          const Dtype* pbox_data, const int num_pboxes, const int num_classes,
          const Dtype overlap_threshold, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedPboxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, pbox_data, num_pboxes, num_classes,
      overlap_threshold, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedPboxGPU(const int nthreads,
          const float* bbox_data, const int num_bboxes, const int num_classes,
          const float overlap_threshold, bool* overlapped_data);
template void ComputeOverlappedPboxGPU(const int nthreads,
          const double* bbox_data, const int num_bboxes, const int num_classes,
          const double overlap_threshold, bool* overlapped_data);

template <typename Dtype>
__global__ void ComputeOverlappedPboxByIdxKernel(const int nthreads,
          const Dtype* pbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int j = index % num_idx;
    const int i = (index / num_idx);
    if (i == j) {
      // Ignore same bbox.
      return;
    }
    // Compute overlap between i-th bbox and j-th bbox.
    const int start_loc_i = idx[i] * 8;
    const int start_loc_j = idx[j] * 8;
    const Dtype overlap = JaccardOverlapPboxGPU<Dtype>(pbox_data + start_loc_i,
        pbox_data + start_loc_j);
    if (overlap > overlap_threshold) {
      overlapped_data[index] = true;
    }
  }
}

template <typename Dtype>
void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const Dtype* pbox_data, const Dtype overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeOverlappedPboxByIdxKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, pbox_data, overlap_threshold,
      idx, num_idx, overlapped_data);
  CUDA_POST_KERNEL_CHECK;
}

template void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const float* pbox_data, const float overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data);
template void ComputeOverlappedPboxByIdxGPU(const int nthreads,
          const double* pbox_data, const double overlap_threshold,
          const int* idx, const int num_idx, bool* overlapped_data); */

template <typename Dtype>
void ApplyNMSGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices) {
  // Keep part of detections whose scores are higher than confidence threshold.
  vector<int> idx;
  vector<Dtype> confidences;
  for (int i = 0; i < num_bboxes; ++i) {
    if (conf_data[i] > confidence_threshold) {
      idx.push_back(i);
      confidences.push_back(conf_data[i]);
    }
  }
  int num_remain = confidences.size();
  if (num_remain == 0) {
    return;
  }
  // Sort detections based on score.
  thrust::sort_by_key(&confidences[0], &confidences[0] + num_remain, &idx[0],
      thrust::greater<Dtype>());
  if (top_k > -1 && top_k < num_remain) {
    num_remain = top_k;
  }

  // Compute overlap between remaining detections.
  Blob<int> idx_blob(1, 1, 1, num_remain);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(idx.begin(), idx.begin() + num_remain, idx_data);

  Blob<bool> overlapped(1, 1, num_remain, num_remain);
  const int total_bboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedByIdxGPU<Dtype>(total_bboxes, bbox_data, nms_threshold,
      idx_blob.gpu_data(), num_remain, overlapped_data);

  // Do non-maximum suppression based on overlapped results.
  const bool* overlapped_results = overlapped.cpu_data();
  vector<int> selected_indices;
  ApplyNMS(overlapped_results, num_remain, &selected_indices);

  // Put back the selected information.
  for (int i = 0; i < selected_indices.size(); ++i) {
    indices->push_back(idx[selected_indices[i]]);
  }
}

template
void ApplyNMSGPU(const float* bbox_data, const float* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);
template
void ApplyNMSGPU(const double* bbox_data, const double* conf_data,
          const int num_bboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);
		  
//******************************************************************************
template <typename Dtype>
void ApplyPboxNMSGPU(const Dtype* pbox_data, const Dtype* conf_data,
          const int num_pboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices) {
  // Keep part of detections whose scores are higher than confidence threshold.
  vector<int> idx;
  vector<Dtype> confidences;
  for (int i = 0; i < num_pboxes; ++i) {
    if (conf_data[i] > confidence_threshold) {
      idx.push_back(i);
      confidences.push_back(conf_data[i]);
    }
  }
  int num_remain = confidences.size();
  if (num_remain == 0) {
    return;
  }
  // Sort detections based on score.
  thrust::sort_by_key(&confidences[0], &confidences[0] + num_remain, &idx[0],
      thrust::greater<Dtype>());
  if (top_k > -1 && top_k < num_remain) {
    num_remain = top_k;
  }

  // Compute overlap between remaining detections.
  Blob<int> idx_blob(1, 1, 1, num_remain);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(idx.begin(), idx.begin() + num_remain, idx_data);

  Blob<bool> overlapped(1, 1, num_remain, num_remain);
  const int total_pboxes = overlapped.count();
  bool* overlapped_data = overlapped.mutable_gpu_data();
  ComputeOverlappedPboxByIdxGPU<Dtype>(total_pboxes, pbox_data, nms_threshold,
      idx_blob.gpu_data(), num_remain, overlapped_data);

  // Do non-maximum suppression based on overlapped results.
  const bool* overlapped_results = overlapped.cpu_data();
  vector<int> selected_indices;
  ApplyPboxNMS(overlapped_results, num_remain, &selected_indices);

  // Put back the selected information.
  for (int i = 0; i < selected_indices.size(); ++i) {
    indices->push_back(idx[selected_indices[i]]);
  }
}

template
void ApplyPboxNMSGPU(const float* pbox_data, const float* conf_data,
          const int num_pboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);
template
void ApplyPboxNMSGPU(const double* pbox_data, const double* conf_data,
          const int num_pboxes, const float confidence_threshold,
          const int top_k, const float nms_threshold, vector<int>* indices);

		
		
template <typename Dtype>
__global__ void GetDetectionsKernel(const int nthreads,
          const Dtype* bbox_data, const Dtype* conf_data, const int image_id,
          const int label, const int* indices, const bool clip_bbox,
          Dtype* detection_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int det_idx = indices[index];
    detection_data[index * 7] = image_id;
    detection_data[index * 7 + 1] = label;
    detection_data[index * 7 + 2] = conf_data[det_idx];
    if (clip_bbox) {
      ClipBBoxGPU(&(bbox_data[det_idx * 4]), &(detection_data[index * 7 + 3]));
    } else {
      for (int i = 0; i < 4; ++i) {
        detection_data[index * 7 + 3 + i] = bbox_data[det_idx * 4 + i];
      }
    }
  }
}

template <typename Dtype>
void GetDetectionsGPU(const Dtype* bbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<Dtype>* detection_blob) {
  // Store selected indices in array.
  int num_det = indices.size();
  if (num_det == 0) {
    return;
  }
  Blob<int> idx_blob(1, 1, 1, num_det);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(indices.begin(), indices.end(), idx_data);
  // Prepare detection_blob.
  detection_blob->Reshape(1, 1, num_det, 7);
  Dtype* detection_data = detection_blob->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GetDetectionsKernel<Dtype><<<CAFFE_GET_BLOCKS(num_det),
      CAFFE_CUDA_NUM_THREADS>>>(num_det, bbox_data, conf_data, image_id, label,
      idx_blob.gpu_data(), clip_bbox, detection_data);
  CUDA_POST_KERNEL_CHECK;
}

template void GetDetectionsGPU(const float* bbox_data, const float* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<float>* detection_blob);
template void GetDetectionsGPU(const double* bbox_data, const double* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_bbox, Blob<double>* detection_blob);
		  
//******************************************************************************
template <typename Dtype>
__global__ void GetPboxDetectionsKernel(const int nthreads,
          const Dtype* pbox_data, const Dtype* conf_data, const int image_id,
          const int label, const int* indices, const bool clip_pbox,
          Dtype* detection_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int det_idx = indices[index];
    detection_data[index * 11] = image_id;
    detection_data[index * 11 + 1] = label;
    detection_data[index * 11 + 2] = conf_data[det_idx];
    if (clip_pbox) {
      ClipBBoxGPU(&(pbox_data[det_idx * 8]), &(detection_data[index * 11 + 3]));
    } else {
      for (int i = 0; i < 8; ++i) {
        detection_data[index * 11 + 3 + i] = pbox_data[det_idx * 8 + i];
      }
    }
  }
}

template <typename Dtype>
void GetPboxDetectionsGPU(const Dtype* pbox_data, const Dtype* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_pbox, Blob<Dtype>* detection_blob) {
  // Store selected indices in array.
  int num_det = indices.size();
  if (num_det == 0) {
    return;
  }
  Blob<int> idx_blob(1, 1, 1, num_det);
  int* idx_data = idx_blob.mutable_cpu_data();
  std::copy(indices.begin(), indices.end(), idx_data);
  // Prepare detection_blob.
  detection_blob->Reshape(1, 1, num_det, 11);
  Dtype* detection_data = detection_blob->mutable_gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GetPboxDetectionsKernel<Dtype><<<CAFFE_GET_BLOCKS(num_det),
      CAFFE_CUDA_NUM_THREADS>>>(num_det, pbox_data, conf_data, image_id, label,
      idx_blob.gpu_data(), clip_pbox, detection_data);
  CUDA_POST_KERNEL_CHECK;
}

template void GetPboxDetectionsGPU(const float* pbox_data, const float* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_pbox, Blob<float>* detection_blob);
template void GetPboxDetectionsGPU(const double* pbox_data, const double* conf_data,
          const int image_id, const int label, const vector<int>& indices,
          const bool clip_pbox, Blob<double>* detection_blob);		  
		  
template <typename Dtype>
__global__ void ComputeConfLossKernel(const int nthreads,
    const Dtype* conf_data, const int num_preds_per_class,
    const int num_classes, const ConfLossType loss_type,
    const Dtype* match_data, Dtype* conf_loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int label = match_data[index];
    int num = index / num_preds_per_class;
    int p = index % num_preds_per_class;
    int start_idx = (num * num_preds_per_class + p) * num_classes;
    Dtype loss = 0;
    if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
      // Compute softmax probability.
      Dtype prob = conf_data[start_idx + label];
      loss = -log(Max(prob, Dtype(FLT_MIN)));
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
    }
    conf_loss_data[index] = loss;
  }
}

template <typename Dtype>
void ComputeConfLossGPU(const Blob<Dtype>& conf_blob, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss) {
  CHECK_LT(background_label_id, num_classes);
  Blob<Dtype> match_blob(num, num_preds_per_class, 1, 1);
  Dtype* match_data = match_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
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
      match_data[i * num_preds_per_class + p] = label;
    }
  }
  // Get probability data.
  const Dtype* conf_gpu_data = conf_blob.gpu_data();
  Blob<Dtype> prob_blob;
  prob_blob.ReshapeLike(conf_blob);
  if (loss_type == MultiBoxLossParameter_ConfLossType_SOFTMAX) {
    Dtype* prob_gpu_data = prob_blob.mutable_gpu_data();
    SoftMaxGPU(conf_blob.gpu_data(), num * num_preds_per_class, num_classes, 1,
        prob_gpu_data);
    conf_gpu_data = prob_blob.gpu_data();
  }
  // Compute the loss.
  Blob<Dtype> conf_loss_blob(num, num_preds_per_class, 1, 1);
  Dtype* conf_loss_gpu_data = conf_loss_blob.mutable_gpu_data();
  const int num_threads = num * num_preds_per_class;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputeConfLossKernel<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
    CAFFE_CUDA_NUM_THREADS>>>(num_threads, conf_gpu_data, num_preds_per_class,
        num_classes, loss_type, match_blob.gpu_data(), conf_loss_gpu_data);
  // Save the loss.
  all_conf_loss->clear();
  const Dtype* loss_data = conf_loss_blob.cpu_data();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss(loss_data, loss_data + num_preds_per_class);
    all_conf_loss->push_back(conf_loss);
    loss_data += num_preds_per_class;
  }
}

// Explicit initialization.
template void ComputeConfLossGPU(const Blob<float>& conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);
template void ComputeConfLossGPU(const Blob<double>& conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const ConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedBBox> >& all_gt_bboxes,
      vector<vector<float> >* all_conf_loss);
	  
//********************************************************************	  
template <typename Dtype>
__global__ void ComputePboxConfLossKernel(const int nthreads,
    const Dtype* conf_data, const int num_preds_per_class,
    const int num_classes, const PConfLossType loss_type,
    const Dtype* match_data, Dtype* conf_loss_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int label = match_data[index];
    int num = index / num_preds_per_class;
    int p = index % num_preds_per_class;
    int start_idx = (num * num_preds_per_class + p) * num_classes;
    Dtype loss = 0;
    if (loss_type == MultiPBoxLossParameter_ConfLossType_SOFTMAX) {
      // Compute softmax probability.
      Dtype prob = conf_data[start_idx + label];
      loss = -log(Max(prob, Dtype(FLT_MIN)));
    } else if (loss_type == MultiPBoxLossParameter_ConfLossType_LOGISTIC) {
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
    }
    conf_loss_data[index] = loss;
  }
}


template <typename Dtype>
void ComputePboxConfLossGPU(const Blob<Dtype>& conf_blob, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const PConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
      vector<vector<float> >* all_conf_loss) {
  CHECK_LT(background_label_id, num_classes);
  Blob<Dtype> match_blob(num, num_preds_per_class, 1, 1);
  Dtype* match_data = match_blob.mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
    const map<int, vector<int> >& match_indices = all_match_indices[i];
    for (int p = 0; p < num_preds_per_class; ++p) {
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
          // A prior can only be matched to one gt pbox.
          break;
        }
      }
      match_data[i * num_preds_per_class + p] = label;
    }
  }
  // Get probability data.
  const Dtype* conf_gpu_data = conf_blob.gpu_data();
  Blob<Dtype> prob_blob;
  prob_blob.ReshapeLike(conf_blob);
  if (loss_type == MultiPBoxLossParameter_ConfLossType_SOFTMAX) {
    Dtype* prob_gpu_data = prob_blob.mutable_gpu_data();
    SoftMaxGPU(conf_blob.gpu_data(), num * num_preds_per_class, num_classes, 1,
        prob_gpu_data);
    conf_gpu_data = prob_blob.gpu_data();
  }
  // Compute the loss.
  Blob<Dtype> conf_loss_blob(num, num_preds_per_class, 1, 1);
  Dtype* conf_loss_gpu_data = conf_loss_blob.mutable_gpu_data();
  const int num_threads = num * num_preds_per_class;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ComputePboxConfLossKernel<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
    CAFFE_CUDA_NUM_THREADS>>>(num_threads, conf_gpu_data, num_preds_per_class,
        num_classes, loss_type, match_blob.gpu_data(), conf_loss_gpu_data);
  // Save the loss.
  all_conf_loss->clear();
  const Dtype* loss_data = conf_loss_blob.cpu_data();
  for (int i = 0; i < num; ++i) {
    vector<float> conf_loss(loss_data, loss_data + num_preds_per_class);
    all_conf_loss->push_back(conf_loss);
    //LOG(WARNING) << "confloss " << conf_loss[1];
    loss_data += num_preds_per_class;
  }
}

// Explicit initialization.
template void ComputePboxConfLossGPU(const Blob<float>& conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const PConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
      vector<vector<float> >* all_conf_loss);
template void ComputePboxConfLossGPU(const Blob<double>& conf_data, const int num,
      const int num_preds_per_class, const int num_classes,
      const int background_label_id, const PConfLossType loss_type,
      const vector<map<int, vector<int> > >& all_match_indices,
      const map<int, vector<NormalizedPBox> >& all_gt_pboxes,
      vector<vector<float> >* all_conf_loss);

}  // namespace caffe
