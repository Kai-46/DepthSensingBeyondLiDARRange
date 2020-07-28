clear all; close all; clc;
format long;

%==========================================================================
%geometric setup of the three cameras
%==========================================================================
% check data/elephant/info_dict.json
back_dist = 20.577952178726964;
baseline = 20.577952178726964;      
focal_length = 43962.93892852579;   % pixels
image_dir = './data/elephant';
out_dir = './output';
CUDA_DEVICE = '0';

if ~exist(out_dir, 'dir')
   mkdir(out_dir)
end

%==========================================================================
%==========================================================================
% pseudo-rectify
%==========================================================================
%==========================================================================
color_I1 = imread([image_dir, '/color_left.png']);
color_I2 = imread([image_dir, '/color_right.png']);

%==========================================================================
% match surf key points
%==========================================================================
I1 = rgb2gray(color_I1); I2 = rgb2gray(color_I2);

points1 = detectSURFFeatures(I1,'MetricThreshold',2000);
points2 = detectSURFFeatures(I2,'MetricThreshold',2000);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2,'MatchThreshold',2.0) ;
left_matched_points = vpts1(indexPairs(:,1));
right_matched_points = vpts2(indexPairs(:,2));

% visualize the matches for debugging purposes
% figure;
% subplot(121);
% imshow(I1); hold on;
% plot(left_matched_points.selectStrongest(100));
% xlabel('Left View');
% subplot(122);
% imshow(I2); hold on;
% plot(right_matched_points.selectStrongest(100));
% xlabel('Right View');
% % sgtitle('Strongest 100 Surf Matches');

left_matched_points = left_matched_points.Location;
right_matched_points = right_matched_points.Location;
fprintf('Matched %i points\n', size(left_matched_points, 1));

%==========================================================================
% estimate 2*3 affine matrices that pseudo-rectify the left, right images
%==========================================================================

scale = 2000;
num_ransc_trials = 5000;
min_set_size = 10;
max_support = 0;
thres = 2;          % error below 2 pixels to be considered as inlier
x_diff = 0;         % the content of pseduo-rectified right image should move left-ward w.r.t left image
for i=1:num_ransc_trials
    % randomly sample the minimum set
    tmp = randperm(size(right_matched_points, 1));
    tmp = tmp(1:min_set_size);
    A = [left_matched_points(tmp, :) / scale, -right_matched_points(tmp, :) / scale, ones(min_set_size, 1)]; % use noisy matches
    
    [U,D,V] = svd(A,0);
    x1 = V(:,end);
    
    % make sure unit norm for the first two components
    x1(1:4) = x1(1:4) / scale;  % numerical trick
    x1 = x1 / norm(x1(1:2));
    % positive sign for a22
    x1 = x1 / sign(x1(2));
    
    % check size of the support set
    tmp = [left_matched_points, -right_matched_points, ones(size(left_matched_points, 1), 1)] * x1;
    mask = abs(tmp) < thres;
    support = sum(mask) / size(tmp, 1);
    fprintf('ransac trial %i, support %.4f\n', i, support);
    if support > max_support
       max_support = support;
       x = x1;
       inlier_mask = mask;
    end
end
fprintf('End of ransac, max_support %.4f\n', max_support);

%==========================================================================
% compose affine matrices for both views
%==========================================================================
% left view
col_vec1 = x(1:2, :);
col_vec2 = [-col_vec1(2);col_vec1(1)];
rot_mat = [col_vec2, col_vec1];
% check determinant
if (det(rot_mat) < 0)
   col_vec2 = -col_vec2;
   rot_mat = [col_vec2, col_vec1];
end
affine_mat_1 = [rot_mat', [0., 0.]'];

% right view
col_vec1 = x(3:4, :);
col_vec2 = [-col_vec1(2);col_vec1(1)];
rot_mat = [col_vec2, col_vec1];
% check determinant
if (det(rot_mat) < 0)
   col_vec2 = -col_vec2;
   rot_mat = [col_vec2, col_vec1];
end
tmp = 0. - x(5, 1);
affine_mat_2 = [rot_mat', [0., tmp]'];

cnt = sum(inlier_mask(:));
x_diff = [right_matched_points(inlier_mask, :), ones(cnt, 1)] * reshape(affine_mat_2(1, :), 3, 1) ...
            - [left_matched_points(inlier_mask, :), ones(cnt, 1)] * reshape(affine_mat_1(1, :), 3, 1);
x_diff = median(x_diff);
margin = 50.0;
x_translation = -(x_diff + margin);
affine_mat_2(1, 3) = x_translation;

disp('Estimated affine matrix for left view:')
disp(affine_mat_1);
disp('Estimated affine matrix for right view:')
disp(affine_mat_2);

%==========================================================================
%apply pseudo-rectification and write rectified pairs
%==========================================================================
pseudo_rectify_dir = [out_dir, '/pseudo_rectify'];
if ~exist(pseudo_rectify_dir, 'dir')
   mkdir(pseudo_rectify_dir)
end

tform = affine2d([affine_mat_1', [0; 0; 1]]);
pseudo_rect_I1 = imwarp_same(color_I1, tform);

tform = affine2d([affine_mat_2', [0; 0; 1]]);
pseudo_rect_I2 = imwarp_same(color_I2, tform);

csvwrite([pseudo_rectify_dir, '/affine_mat_im0.txt'], affine_mat_1);
imwrite(color_I1, [pseudo_rectify_dir, '/orig_im0.png']);
imwrite(color_I2, [pseudo_rectify_dir, '/orig_im1.png']);
imwrite(pseudo_rect_I1, [pseudo_rectify_dir, '/im0.png']);
imwrite(pseudo_rect_I2, [pseudo_rectify_dir, '/im1.png']);

%==========================================================================
% create a small area to visually inspect the quality of rectification
%==========================================================================
figure;
subplot(121);
imshow(pseudo_rect_I1(1822:1922, 1871:1971, :));
title('Crop of Rectified Left view');
subplot(122);
imshow(pseudo_rect_I2(1822:1922, 1851:1951, :));
title('Crop of Rectified Right view');
set(gcf,'color','w');

%==========================================================================
%==========================================================================
% run stereo matching
%==========================================================================
%==========================================================================
tmp_dir = [pseudo_rectify_dir, '/tmp'];
if ~exist(tmp_dir, 'dir')
   mkdir(tmp_dir)
end
cmd = ['cp ' pseudo_rectify_dir, '/im0.png ', tmp_dir];
system(cmd);
cmd = ['cp ' pseudo_rectify_dir, '/im1.png ', tmp_dir];
system(cmd);

disp_esti_dir = [out_dir, '/disp_esti'];
if ~exist(disp_esti_dir, 'dir')
   mkdir(disp_esti_dir)
end

cmd = ['CUDA_VISIBLE_DEVICES=' CUDA_DEVICE ...
       ' python3 high-res-stereo/submission.py ' ...
       ' --datapath ' pseudo_rectify_dir ...
       ' --outdir ' disp_esti_dir ...
       ' --loadmodel high-res-stereo/final-768px.pth ' ...
       ' --testres 0.5 --clean 1.0 --max_disp 512 '];
disp(cmd);
[status,cmdout] = system(cmd, '-echo');

cmd = ['mv ' disp_esti_dir '/tmp/* ' disp_esti_dir];
system(cmd);
cmd = ['rmdir ' disp_esti_dir '/tmp'];
system(cmd);

system(['rm ' tmp_dir '/*']);
system(['rmdir ' tmp_dir]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%==========================================================================
% ambiguity removal
%==========================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==========================================================================
%load left and back images, and the predicted disparity map
%==========================================================================
ambi_remove_dir = [out_dir, '/ambiguity_remove'];
if ~exist(ambi_remove_dir, 'dir')
   mkdir(ambi_remove_dir)
end

color_I1 = imread([pseudo_rectify_dir, '/im0.png']);
color_I2 = imread([image_dir, '/color_back.png']);
I1 = rgb2gray(color_I1); I2 = rgb2gray(color_I2);
disparity = hdf5read([disp_esti_dir, '/disp.h5'],'data');
disparity = disparity';   

invalid_mask = I1 < 1e-5;   % black pixels

figure;
mask_tmp = invalid_mask | isnan(disparity);
disparity_tmp = disparity;
disparity_tmp(mask_tmp) = max(disparity(~mask_tmp));
imagesc(disparity_tmp,'AlphaData',~mask_tmp);
title('Estimated Disparity');
set(gcf,'color','w');
colorbar('southoutside');

imwrite( ind2rgb(im2uint8(mat2gray(disparity_tmp)), parula(256)), [ambi_remove_dir, '/disp_esti.png'], 'png', 'Alpha', uint8(~mask_tmp) * 255);

%==========================================================================
% match surf key points between left and back images
%==========================================================================
points1 = detectSURFFeatures(I1,'MetricThreshold',2000);
points2 = detectSURFFeatures(I2,'MetricThreshold',2000);

[f1,vpts1] = extractFeatures(I1,points1);
[f2,vpts2] = extractFeatures(I2,points2);

indexPairs = matchFeatures(f1,f2,'MatchThreshold',2.0) ;
forward_matched_points = vpts1(indexPairs(:,1));
backward_matched_points = vpts2(indexPairs(:,2));

forward_matched_points = forward_matched_points.Location;
backward_matched_points = backward_matched_points.Location;

%==========================================================================
% now try to estimate the horizontal ambiguity of disparity
%==========================================================================
max_trials = 5000;
ambiguity = zeros(1, max_trials);
idx = 1;
while idx <= max_trials
    % sample two pixels
    ii = randi(size(forward_matched_points, 1));
    while 1
        jj = randi(size(forward_matched_points, 1));
        if jj ~= ii
            break;
        end
    end
    
    % check their disparity
    forward_ii_x = forward_matched_points(ii, 1);
    forward_ii_y = forward_matched_points(ii, 2);
    forward_jj_x = forward_matched_points(jj, 1);
    forward_jj_y = forward_matched_points(jj, 2);
    
    backward_ii_x = backward_matched_points(ii, 1);
    backward_ii_y = backward_matched_points(ii, 2);
    backward_jj_x = backward_matched_points(jj, 1);
    backward_jj_y = backward_matched_points(jj, 2);
    
    forward_ii_x = round(forward_ii_x);
    forward_ii_y = round(forward_ii_y);
    forward_jj_x = round(forward_jj_x);
    forward_jj_y = round(forward_jj_y);
    
    backward_ii_x = round(backward_ii_x);
    backward_ii_y = round(backward_ii_y );
    backward_jj_x = round(backward_jj_x);
    backward_jj_y = round(backward_jj_y);
    
    % check the mask
    if invalid_mask(forward_ii_y, forward_ii_x) || invalid_mask(forward_jj_y, forward_jj_x)
        continue;
    end
    
    % check the disparity
    disp1 = disparity(forward_ii_y,forward_ii_x);
    disp2 = disparity(forward_jj_y, forward_jj_x);
    thres = 5;
    if(isnan(disp1) ||...
       isnan(disp2) ||...
       abs(disp1 - disp2) > thres)
       continue;
    end
    
    % check pixel distance
    forward_dist = sqrt((forward_ii_x - forward_jj_x).^2 + ...
                        (forward_ii_y - forward_jj_y).^2);
    backward_dist = sqrt((backward_ii_x - backward_jj_x).^2 + ...
                        (backward_ii_y - backward_jj_y).^2);
    thres = 200;
    if (forward_dist < thres || forward_dist < backward_dist)
        continue;
    end
    
    % compute expected disparity
    expected_disp = focal_length * baseline / back_dist * (forward_dist / backward_dist - 1);
    diff = expected_disp - (disp1 + disp2) / 2;
    
    clip_thres = 1000;
    if (diff < 0 || diff > clip_thres)
       continue; 
    end
    
    ambiguity(1, idx) = diff;
    
    fprintf('Trial %i, Ambiguity %f\n', idx, diff);
    
    idx = idx + 1;
end

adjust = median(ambiguity);
fprintf('Median Ambiguity %f\n', adjust);

figure;
ambiguity(ambiguity > clip_thres) = clip_thres;
ambiguity(ambiguity < -clip_thres) = -clip_thres;
h = histogram(ambiguity);
hold on; 
line([adjust, adjust], [0, max(h.Values) + 100], 'Color', 'r', 'LineWidth', 2);
ylim([0, max(h.Values) + 100]);
title('Distribution of All Cached Ambiguity Estimates');
set(gcf,'color','w');

%==========================================================================
% add estimated ambiguity to the predicted disparity
% and convert disparity to depth
%==========================================================================
disparity = disparity + adjust;
esti_depth = focal_length * baseline ./ disparity;

%==========================================================================
% visualize results
%==========================================================================
figure;
nan_mask = isnan(esti_depth) | invalid_mask;

mask_tmp = nan_mask;
esti_depth_tmp = esti_depth;
clip_min = 200;
clip_max = 260;
esti_depth_tmp(esti_depth_tmp < clip_min) = clip_min;
esti_depth_tmp(esti_depth_tmp > clip_max) = clip_max;
imagesc(esti_depth_tmp,'AlphaData', ~nan_mask);
title('Estimated depth');
set(gcf,'color','w');
colorbar('southoutside');

imwrite(color_I1, [ambi_remove_dir, '/left_view.png']);
imwrite( ind2rgb(im2uint8(mat2gray(esti_depth_tmp)), parula(256)), [ambi_remove_dir, '/depth_esti.png'], 'png', 'Alpha', uint8(~nan_mask) * 255);

%==========================================================================





