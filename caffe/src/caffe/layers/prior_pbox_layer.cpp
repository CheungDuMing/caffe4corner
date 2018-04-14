#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <cmath>
#include "caffe/layers/prior_pbox_layer.hpp"

#define pi 3.14159265359

namespace caffe {

	template <typename Dtype>
	void PriorPBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const PriorPBoxParameter& prior_pbox_param =
			this->layer_param_.prior_pbox_param();
        num_pboxes_rotation = prior_pbox_param.num_pboxes();
		CHECK_GT(prior_pbox_param.min_size_size(), 0) << "must provide min_size.";
		for (int i = 0; i < prior_pbox_param.min_size_size(); ++i) {
			min_sizes_.push_back(prior_pbox_param.min_size(i));
			CHECK_GT(min_sizes_.back(), 0) << "min_size must be positive.";
		}
		aspect_ratios_.clear();
		aspect_ratios_.push_back(1.);
		flip_ = prior_pbox_param.flip();
		for (int i = 0; i < prior_pbox_param.aspect_ratio_size(); ++i) {
			float ar = prior_pbox_param.aspect_ratio(i);
			bool already_exist = false;
			for (int j = 0; j < aspect_ratios_.size(); ++j) {
				if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
					already_exist = true;
					break;
				}
			}
			if (!already_exist) {
				aspect_ratios_.push_back(ar);
				if (flip_) {
					aspect_ratios_.push_back(1. / ar);
				}
			}
		}
		num_priors_ = aspect_ratios_.size() * min_sizes_.size() * num_pboxes_rotation;
		if (prior_pbox_param.max_size_size() > 0) {
			CHECK_EQ(prior_pbox_param.min_size_size(), prior_pbox_param.max_size_size());
			for (int i = 0; i < prior_pbox_param.max_size_size(); ++i) {
				max_sizes_.push_back(prior_pbox_param.max_size(i));
				CHECK_GT(max_sizes_[i], min_sizes_[i])
					<< "max_size must be greater than min_size.";
				num_priors_ += num_pboxes_rotation;//4;
			}
		}
		clip_ = prior_pbox_param.clip();
		if (prior_pbox_param.variance_size() > 1) {
			// Must and only provide 4 variance.
			//*************************************************
			//change to "Must and only provide 8 variance"
			CHECK_EQ(prior_pbox_param.variance_size(), 8);
			for (int i = 0; i < prior_pbox_param.variance_size(); ++i) {
				CHECK_GT(prior_pbox_param.variance(i), 0);
				variance_.push_back(prior_pbox_param.variance(i));
			}
		}
		else if (prior_pbox_param.variance_size() == 1) {
			CHECK_GT(prior_pbox_param.variance(0), 0);
			variance_.push_back(prior_pbox_param.variance(0));
		}
		else {
			// Set default to 0.1.
			variance_.push_back(0.1);
		}

		if (prior_pbox_param.has_img_h() || prior_pbox_param.has_img_w()) {
			CHECK(!prior_pbox_param.has_img_size())
				<< "Either img_size or img_h/img_w should be specified; not both.";
			img_h_ = prior_pbox_param.img_h();
			CHECK_GT(img_h_, 0) << "img_h should be larger than 0.";
			img_w_ = prior_pbox_param.img_w();
			CHECK_GT(img_w_, 0) << "img_w should be larger than 0.";
		}
		else if (prior_pbox_param.has_img_size()) {
			const int img_size = prior_pbox_param.img_size();
			CHECK_GT(img_size, 0) << "img_size should be larger than 0.";
			img_h_ = img_size;
			img_w_ = img_size;
		}
		else {
			img_h_ = 0;
			img_w_ = 0;
		}

		if (prior_pbox_param.has_step_h() || prior_pbox_param.has_step_w()) {
			CHECK(!prior_pbox_param.has_step())
				<< "Either step or step_h/step_w should be specified; not both.";
			step_h_ = prior_pbox_param.step_h();
			CHECK_GT(step_h_, 0.) << "step_h should be larger than 0.";
			step_w_ = prior_pbox_param.step_w();
			CHECK_GT(step_w_, 0.) << "step_w should be larger than 0.";
		}
		else if (prior_pbox_param.has_step()) {
			const float step = prior_pbox_param.step();
			CHECK_GT(step, 0) << "step should be larger than 0.";
			step_h_ = step;
			step_w_ = step;
		}
		else {
			step_h_ = 0;
			step_w_ = 0;
		}

		offset_ = prior_pbox_param.offset();
	}

	template <typename Dtype>
	void PriorPBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int layer_width = bottom[0]->width();
		const int layer_height = bottom[0]->height();
		vector<int> top_shape(3, 1);
		// Since all images in a batch has same height and width, we only need to
		// generate one set of priors which can be shared across all images.
		top_shape[0] = 1;
		// 2 channels. First channel stores the mean of each prior coordinate.
		// Second channel stores the variance of each prior coordinate.
		top_shape[1] = 2;
		top_shape[2] = layer_width * layer_height * num_priors_ * 8;
		CHECK_GT(top_shape[2], 0);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void PriorPBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int layer_width = bottom[0]->width();
		const int layer_height = bottom[0]->height();
		int img_width, img_height;
		if (img_h_ == 0 || img_w_ == 0) {
			//*************************************************//
			// because in the train.prototxt
			// bottom: "conv9_2"
			// bottom : "data"
			img_width = bottom[1]->width();
			img_height = bottom[1]->height();
		}
		else {
			img_width = img_w_;
			img_height = img_h_;
		}
		float step_w, step_h;
		if (step_w_ == 0 || step_h_ == 0) {
			step_w = static_cast<float>(img_width) / layer_width;
			step_h = static_cast<float>(img_height) / layer_height;
		}
		else {
			step_w = step_w_;
			step_h = step_h_;
		}
		Dtype* top_data = top[0]->mutable_cpu_data();
		int dim = layer_height * layer_width * num_priors_ * 8;
		int idx = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				//***************************************************************
				// need new approach to generate prior pboxes
				// step : get the center with respect to real image
				float center_x = (w + offset_) * step_w;
				float center_y = (h + offset_) * step_h;
				//float pbox_width, pbox_height;
				float temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
				for (int s = 0; s < min_sizes_.size(); ++s) {
					int min_size_ = min_sizes_[s];
					// first prior: aspect_ratio = 1, size = min_size
					box_width = box_height = min_size_;
					// ltopx
					top_data[idx++] = temp1 = (center_x - box_width / 2.) / img_width;
					// ltopy
					top_data[idx++] = temp2 = (center_y - box_height / 2.) / img_height;
					//lbottomx
					top_data[idx++] = temp3 = (center_x - box_width / 2.) / img_width;
					//lbottomy
					top_data[idx++] = temp4 = (center_y + box_height / 2.) / img_height;
					// rbottomx
					top_data[idx++] = temp5 = (center_x + box_width / 2.) / img_width;
					// rbottomy
					top_data[idx++] = temp6 = (center_y + box_height / 2.) / img_height;
					// rtopx
					top_data[idx++] = temp7 = (center_x + box_width / 2.) / img_width;
					// rtopy
					top_data[idx++] = temp8 = (center_y - box_height / 2.) / img_height;

                    if(num_pboxes_rotation == 4) {
                    //4 pboxes in spaces
                    for (float i = 1. ; i < 3. ; i = i + 1. )
                    {
                        // ltopx
                        top_data[idx++] = temp1 * cos(i* pi / 2. ) + temp2 * sin(i* pi / 2. );
                        // ltopy
                        top_data[idx++] = temp2 * cos(i * pi / 2. ) - temp1 * sin(i* pi / 2. );
                        //lbottomx
                        top_data[idx++] = temp3 * cos(i* pi / 2. ) + temp4 * sin(i* pi / 2. );
                        //lbottomy
                        top_data[idx++] = temp4 * cos(i * pi / 2. ) - temp3 * sin(i* pi / 2. );
                        // rbottomx
                        top_data[idx++] = temp5 * cos(i* pi / 2. ) + temp6 * sin(i* pi / 2. );
                        // rbottomy
                        top_data[idx++] = temp6 * cos(i * pi / 2. ) - temp5 * sin(i* pi / 2. );
                        // rtopx
                        top_data[idx++] = temp7 * cos(i* pi / 2. ) + temp8 * sin(i* pi / 2. );
                        // rtopy
                        top_data[idx++] = temp8 * cos(i * pi / 2. ) - temp7 * sin(i* pi / 2. );
                    }

                    for (float j = 3. ; j < 4. ; j = j + 1. )
                    {
                        float i;
                        i = 4. - j;
                        // ltopx
                        top_data[idx++] = temp1 * cos(i* pi / 2. ) - temp2 * sin(i* pi / 2. );
                        // ltopy
                        top_data[idx++] = temp2 * cos(i * pi / 2. ) + temp1 * sin(i* pi / 2. );
                        //lbottomx
                        top_data[idx++] = temp3 * cos(i* pi / 2. ) - temp4 * sin(i* pi / 2. );
                        //lbottomy
                        top_data[idx++] = temp4 * cos(i * pi / 2. ) + temp3 * sin(i* pi / 2. );
                        // rbottomx
                        top_data[idx++] = temp5 * cos(i* pi / 2. ) - temp6 * sin(i* pi / 2. );
                        // rbottomy
                        top_data[idx++] = temp6 * cos(i * pi / 2. ) + temp5 * sin(i* pi / 2. );
                        // rtopx
                        top_data[idx++] = temp7 * cos(i* pi / 2. ) - temp8 * sin(i* pi / 2. );
                        // rtopy
                        top_data[idx++] = temp8 * cos(i * pi / 2. ) + temp7 * sin(i* pi / 2. );
                    }
                    }
                    if(num_pboxes_rotation == 12) {
                    //12 pboxes in spaces
                    for (float i = 1. ; i < 7. ; i = i + 1. )
                    {
                        // ltopx
                        top_data[idx++] = temp1 * cos(i* pi / 6. ) + temp2 * sin(i* pi / 6. );
                        // ltopy
                        top_data[idx++] = temp2 * cos(i * pi / 6. ) - temp1 * sin(i* pi / 6. );
                        //lbottomx
                        top_data[idx++] = temp3 * cos(i* pi / 6. ) + temp4 * sin(i* pi / 6. );
                        //lbottomy
                        top_data[idx++] = temp4 * cos(i * pi / 6. ) - temp3 * sin(i* pi / 6. );
                        // rbottomx
                        top_data[idx++] = temp5 * cos(i* pi / 6. ) + temp6 * sin(i* pi / 6. );
                        // rbottomy
                        top_data[idx++] = temp6 * cos(i * pi / 6. ) - temp5 * sin(i* pi / 6. );
                        // rtopx
                        top_data[idx++] = temp7 * cos(i* pi / 6. ) + temp8 * sin(i* pi / 6. );
                        // rtopy
                        top_data[idx++] = temp8 * cos(i * pi / 6. ) - temp7 * sin(i* pi / 6. );
                    }

                    for (float j = 7. ; j < 12. ; j = j + 1. )
                    {
                        float i;
                        i = 12. - j;
                        // ltopx
                        top_data[idx++] = temp1 * cos(i* pi / 6. ) - temp2 * sin(i* pi / 6. );
                        // ltopy
                        top_data[idx++] = temp2 * cos(i * pi / 6. ) + temp1 * sin(i* pi / 6. );
                        //lbottomx
                        top_data[idx++] = temp3 * cos(i* pi / 6. ) - temp4 * sin(i* pi / 6. );
                        //lbottomy
                        top_data[idx++] = temp4 * cos(i * pi / 6. ) + temp3 * sin(i* pi / 6. );
                        // rbottomx
                        top_data[idx++] = temp5 * cos(i* pi / 6. ) - temp6 * sin(i* pi / 6. );
                        // rbottomy
                        top_data[idx++] = temp6 * cos(i * pi / 6. ) + temp5 * sin(i* pi / 6. );
                        // rtopx
                        top_data[idx++] = temp7 * cos(i* pi / 6. ) - temp8 * sin(i* pi / 6. );
                        // rtopy
                        top_data[idx++] = temp8 * cos(i * pi / 6. ) + temp7 * sin(i* pi / 6. );
                    }
                    }

					if (max_sizes_.size() > 0) {
						CHECK_EQ(min_sizes_.size(), max_sizes_.size());
						int max_size_ = max_sizes_[s];
						// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
						box_width = box_height = sqrt(min_size_ * max_size_);
						// ltopx
						top_data[idx++] = temp1 = (center_x - box_width / 2.) / img_width;
						// ltopy
						top_data[idx++] = temp2 = (center_y - box_height / 2.) / img_height;
						//lbottomx
						top_data[idx++] = temp3 = (center_x + box_width / 2.) / img_width;
						//lbottomy
						top_data[idx++] = temp4 = (center_y - box_height / 2.) / img_height;
						// rbottomx
						top_data[idx++] = temp5 = (center_x + box_width / 2.) / img_width;
						// rbottomy
						top_data[idx++] = temp6 = (center_y + box_height / 2.) / img_height;
						// rtopx
						top_data[idx++] = temp7 = (center_x + box_width / 2.) / img_width;
						// rtopy
						top_data[idx++] = temp8 = (center_y - box_height / 2.) / img_height;

                        if(num_pboxes_rotation == 4) {
						//4 pboxes in spaces
						for (float i = 1.0; i < 3.f; i = i + 1.f)
                        {
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 2.f) + temp2 * sin(i* pi / 2.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 2.f) - temp1 * sin(i* pi / 2.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 2.f) + temp4 * sin(i* pi / 2.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 2.f) - temp3 * sin(i* pi / 2.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 2.f) + temp6 * sin(i* pi / 2.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 2.f) - temp5 * sin(i* pi / 2.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 2.f) + temp8 * sin(i* pi / 2.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 2.f) - temp7 * sin(i* pi / 2.f);
                        }

                        for (float j = 3.f; j < 4.f; j = j + 1.f)
                        {
                            float i;
                            i = 4.f - j;
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 2.f) - temp2 * sin(i* pi / 2.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 2.f) + temp1 * sin(i* pi / 2.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 2.f) - temp4 * sin(i* pi / 2.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 2.f) + temp3 * sin(i* pi / 2.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 2.f) - temp6 * sin(i* pi / 2.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 2.f) + temp5 * sin(i* pi / 2.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 2.f) - temp8 * sin(i* pi / 2.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 2.f) + temp7 * sin(i* pi / 2.f);
                        }
                        }

                        if(num_pboxes_rotation == 12) {
                        for (float i = 1.0; i < 7.f; i = i + 1.f)
                        {
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 6.f) + temp2 * sin(i* pi / 6.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 6.f) - temp1 * sin(i* pi / 6.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 6.f) + temp4 * sin(i* pi / 6.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 6.f) - temp3 * sin(i* pi / 6.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 6.f) + temp6 * sin(i* pi / 6.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 6.f) - temp5 * sin(i* pi / 6.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 6.f) + temp8 * sin(i* pi / 6.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 6.f) - temp7 * sin(i* pi / 6.f);
                        }

                        for (float j = 7.f; j < 12.f; j = j + 1.f)
                        {
                            float i;
                            i = 12.f - j;
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 6.f) - temp2 * sin(i* pi / 6.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 6.f) + temp1 * sin(i* pi / 6.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 6.f) - temp4 * sin(i* pi / 6.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 6.f) + temp3 * sin(i* pi / 6.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 6.f) - temp6 * sin(i* pi / 6.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 6.f) + temp5 * sin(i* pi / 6.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 6.f) - temp8 * sin(i* pi / 6.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 6.f) + temp7 * sin(i* pi / 6.f);
                        }
                    }
                    }

					// rest of priors
					for (int r = 0; r < aspect_ratios_.size(); ++r) {
						float ar = aspect_ratios_[r];
						if (fabs(ar - 1.) < 1e-6) {
							continue;
						}
						box_width = min_size_ * sqrt(ar);
						box_height = min_size_ / sqrt(ar);

						// ltopx
						top_data[idx++] = temp1 = (center_x - box_width / 2.) / img_width;
						// ltopy
						top_data[idx++] = temp2 = (center_y - box_height / 2.) / img_height;
						//lbottomx
						top_data[idx++] = temp3 = (center_x + box_width / 2.) / img_width;
						//lbottomy
						top_data[idx++] = temp4 = (center_y - box_height / 2.) / img_height;
						// rbottomx
						top_data[idx++] = temp5 = (center_x + box_width / 2.) / img_width;
						// rbottomy
						top_data[idx++] = temp6 = (center_y + box_height / 2.) / img_height;
						// rtopx
						top_data[idx++] = temp7 = (center_x + box_width / 2.) / img_width;
						// rtopy
						top_data[idx++] = temp8 = (center_y - box_height / 2.) / img_height;

                        if(num_pboxes_rotation == 4) {
						//4 pboxes in spaces
						for (float i = 1.0; i < 3.f; i = i + 1.f)
                        {
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 2.f) + temp2 * sin(i* pi / 2.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 2.f) - temp1 * sin(i* pi / 2.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 2.f) + temp4 * sin(i* pi / 2.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 2.f) - temp3 * sin(i* pi / 2.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 2.f) + temp6 * sin(i* pi / 2.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 2.f) - temp5 * sin(i* pi / 2.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 2.f) + temp8 * sin(i* pi / 2.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 2.f) - temp7 * sin(i* pi / 2.f);
                        }

                        for (float j = 3.f; j < 4.f; j = j + 1.f)
                        {
                            float i;
                            i = 4.f - j;
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 2.f) - temp2 * sin(i* pi / 2.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 2.f) + temp1 * sin(i* pi / 2.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 2.f) - temp4 * sin(i* pi / 2.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 2.f) + temp3 * sin(i* pi / 2.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 2.f) - temp6 * sin(i* pi / 2.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 2.f) + temp5 * sin(i* pi / 2.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 2.f) - temp8 * sin(i* pi / 2.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 2.f) + temp7 * sin(i* pi / 2.f);
                        }
                        }


                        if(num_pboxes_rotation == 12) {
                        for (float i = 1.0; i < 7.f; i = i + 1.f)
                        {
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 6.f) + temp2 * sin(i* pi / 6.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 6.f) - temp1 * sin(i* pi / 6.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 6.f) + temp4 * sin(i* pi / 6.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 6.f) - temp3 * sin(i* pi / 6.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 6.f) + temp6 * sin(i* pi / 6.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 6.f) - temp5 * sin(i* pi / 6.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 6.f) + temp8 * sin(i* pi / 6.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 6.f) - temp7 * sin(i* pi / 6.f);
                        }

                        for (float j = 7.f; j < 12.f; j = j + 1.f)
                        {
                            float i;
                            i = 12.f - j;
                            // ltopx
                            top_data[idx++] = temp1 * cos(i* pi / 6.f) - temp2 * sin(i* pi / 6.f);
                            // ltopy
                            top_data[idx++] = temp2 * cos(i * pi / 6.f) + temp1 * sin(i* pi / 6.f);
                            //lbottomx
                            top_data[idx++] = temp3 * cos(i* pi / 6.f) - temp4 * sin(i* pi / 6.f);
                            //lbottomy
                            top_data[idx++] = temp4 * cos(i * pi / 6.f) + temp3 * sin(i* pi / 6.f);
                            // rbottomx
                            top_data[idx++] = temp5 * cos(i* pi / 6.f) - temp6 * sin(i* pi / 6.f);
                            // rbottomy
                            top_data[idx++] = temp6 * cos(i * pi / 6.f) + temp5 * sin(i* pi / 6.f);
                            // rtopx
                            top_data[idx++] = temp7 * cos(i* pi / 6.f) - temp8 * sin(i* pi / 6.f);
                            // rtopy
                            top_data[idx++] = temp8 * cos(i * pi / 6.f) + temp7 * sin(i* pi / 6.f);
                        }
                        }
                    }
                }
            }
        }
		//LOG(INFO) << "success";
		// clip the prior's coordidate such that it is within [0, 1]
		if (clip_) {
			for (int d = 0; d < dim; ++d) {
				top_data[d] = std::min<Dtype>(std::max<Dtype>(top_data[d], 0.), 1.);
			}
		}
		// set the variance.
		top_data += top[0]->offset(0, 1);
		if (variance_.size() == 1) {
			caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
		}
		else {
			int count = 0;
			for (int h = 0; h < layer_height; ++h) {
				for (int w = 0; w < layer_width; ++w) {
					for (int i = 0; i < num_priors_ ; ++i) {
						//***********************************************
						//need change to j < 8
						for (int j = 0; j < 8; ++j) {
							top_data[count] = variance_[j];
							++count;
						}
					}
				}
			}
		}
	}

	INSTANTIATE_CLASS(PriorPBoxLayer);
	REGISTER_LAYER_CLASS(PriorPBox);

}  // namespace caffe
