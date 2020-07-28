clear all; close all;

% generate scene points randomly
Xmin = -10; Xmax = 10;
Ymin = -10; Ymax = 10;
Zmin = 300; Zmax = 330;
num_points = 300;

X = rand(num_points, 1) * (Xmax - Xmin) + Xmin;
Y = rand(num_points, 1) * (Ymax - Ymin) + Ymin;
Z = rand(num_points, 1) * (Zmax - Zmin) + Zmin;

scene_points = [X,Y,Z,ones(num_points, 1)];

% create cameras
fx = 6.1459e+04; fy = fx;
width = 4608; height = 3456;
cx = width / 2; cy = height / 2;
s = 0;

intrinsic = [fx, 0, 0; s, fy, 0; cx, cy, 1];
cameraParams = cameraParameters('IntrinsicMatrix',intrinsic);

% small baseline setting
rotation_angles = [deg2rad(0), deg2rad(0), deg2rad(0)];    % rotation sequence is ZYX
baseline = 2;

% wide baseline setting
% rotation_angles = [deg2rad(0), deg2rad(-5), deg2rad(0)];    % rotation sequence is ZYX
% baseline = 30;

rotation_mat = EulAng2RotMat(rotation_angles, '213');
translation_vec = [-baseline, 0, 0];

% compute ground-truth essential matrix and fundamental matrix
tvec = translation_vec * rotation_mat';
tmp = [0, tvec(3),-tvec(2);...
       -tvec(3),0,tvec(1);
       tvec(2),-tvec(1),0];
E_gt = tmp * rotation_mat;
E_gt = E_gt ./ max(max(abs(E_gt)));

intrinsic_inv = inv(intrinsic);
F_gt = intrinsic_inv * E_gt * intrinsic_inv';
F_gt = F_gt ./ max(max(abs(F_gt)));

cam_center_gt1 = zeros(1, 3);
cam_center_gt2 = -translation_vec * rotation_mat';

% project to first camera
tmp = scene_points * [eye(3); zeros(1, 3)] * intrinsic;
col_first_cam = tmp(:, 1) ./ tmp(:, 3);
row_first_cam = tmp(:, 2) ./ tmp(:, 3);
% project to second camera
tmp = scene_points * [rotation_mat; translation_vec] * intrinsic;
col_second_cam = tmp(:, 1) ./ tmp(:, 3);
row_second_cam = tmp(:, 2) ./ tmp(:, 3);

% valid mask
valid_mask = col_first_cam >= 0;
valid_mask = valid_mask & (row_first_cam >= 0);
valid_mask = valid_mask & (col_first_cam < width);
valid_mask = valid_mask & (row_first_cam < height);

valid_mask = valid_mask & (col_second_cam >= 0);
valid_mask = valid_mask & (row_second_cam >= 0);
valid_mask = valid_mask & (col_second_cam < width);
valid_mask = valid_mask & (row_second_cam < height);

col_first_cam = col_first_cam(valid_mask);
row_first_cam = row_first_cam(valid_mask);
col_second_cam = col_second_cam(valid_mask);
row_second_cam = row_second_cam(valid_mask);

scene_points = scene_points(valid_mask, :);
raw_pixels1 = [col_first_cam, row_first_cam];
raw_pixels2 = [col_second_cam, row_second_cam];

% check E_gt
errors = zeros(size(raw_pixels1, 1), 1);
for i=1:size(raw_pixels1, 1)
    point = scene_points(i, 1:3);
    tmp = point * E_gt * (point * rotation_mat + translation_vec)';
    errors(i, 1) = abs(tmp);
end
fprintf('max E_gt error on noiseless scene points: %.17f\n', max(max(errors)));

% check whether F_gt is correct
epipolar_err = zeros(size(raw_pixels1, 1), 1);
for i = 1:size(raw_pixels1, 1)
    % geometric error in image 1
    pixel1 = [raw_pixels1(i,:), 1];
    pixel2 = [raw_pixels2(i,:), 1];
    
    % pixel1 * F * pixel2' = 0
    l = pixel2 * F_gt';
    d1 = abs(l * pixel1')/sqrt(l(1,1)*l(1,1)+l(1, 2)*l(1,2));
    
    l = pixel1 * F_gt;
    d2 = abs(l * pixel2')/sqrt(l(1,1)*l(1,1)+l(1,2)*l(1,2));
    
    epipolar_err(i, 1) = (d1 + d2) / 2;
end
fprintf('max F_gt error on noiseless correspondences: %.17f\n', max(max(epipolar_err)));

% triangulation angles
tri_angles_gt = zeros(size(scene_points, 1), 1);
for i=1:size(scene_points, 1)
    point = scene_points(i, 1:3);
    vec1 = cam_center_gt1 - point;
    vec2 = cam_center_gt2 - point;
    tri_angles_gt(i, 1) = rad2deg(vector_angle(vec1', vec2'));
end

% now add some gaussian noise to pixel coordinates
mu = [0, 0];
noise_level = 0.5;
sigma = [noise_level, 0; 0, noise_level];
% rng('default')  % For reproducibility
noise = mvnrnd(mu,sigma,size(raw_pixels2, 1));

matchedPoints1 = raw_pixels1;
matchedPoints2 = raw_pixels2 + noise;

% plot result
figure;
subplot(151);
scatter3(scene_points(:,1),scene_points(:,2),scene_points(:,3));
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Scene Points');

subplot(152);
histogram(tri_angles_gt);
title('Ground-truth Triangulation Angles (degrees)');

subplot(153);
scatter(raw_pixels1(:, 1), raw_pixels1(:, 2));
xlabel('X'); ylabel('Y');
axis equal;
xlim([0, width]);
ylim([0, height]);
title('Left Image');
set(gca,'YDir','reverse')

subplot(154);
scatter(raw_pixels2(:, 1), raw_pixels2(:, 2));
hold on;
scatter(matchedPoints2(:, 1), matchedPoints2(:, 2), '+');
xlabel('X'); ylabel('Y');
axis equal;
xlim([0, width]);
ylim([0, height]);
title('Right Image');
legend('Noiseless', 'Noisy');
set(gca,'YDir','reverse')

subplot(155);
% noise = mvnrnd(mu,sigma,10000);
scatter(noise(:, 1), noise(:, 2));
hold on;
th = 0:pi/50:2*pi;
xunit = 1.0 * cos(th) + 0.0;
yunit = 1.0 * sin(th) + 0.0;
h = plot(xunit, yunit);
axis equal;
axis square;
title('Random Noise Added to Right Image');

sgtitle('Problem Setup');

% perturb on ground-truth rotation angles and translation vector
noise_level = 0.3;  % degrees
gt_eul = RotMat2EulAng(rotation_mat, '213');

init_eul = gt_eul + randn(1, 3) * noise_level;
initRotationMatrix = EulAng2RotMat(init_eul, '213');

noise_level = 0.3;  % meters
initTranslationVector = translation_vec + randn(1, 3) * noise_level;

% bundle adjustment
% see https://www.mathworks.com/help/vision/ref/cameraposetoextrinsics.html
% for matlab's definition of rotation, translation, orientaion, location
camMatrix1 = cameraMatrix(cameraParams,eye(3),zeros(1, 3));
camMatrix2 = cameraMatrix(cameraParams,initRotationMatrix,initTranslationVector);
[init_xyz_points,reprojectionErrors] = triangulate(matchedPoints1,matchedPoints2,camMatrix1,camMatrix2);

ViewId = [uint32(1);uint32(2)]; Orientation={eye(3); initRotationMatrix'}; Location={zeros(1,3); -initTranslationVector * initRotationMatrix'};
init_poses = table(ViewId, Orientation, Location);

for i = 1:size(matchedPoints1, 1)
    points = [matchedPoints1(i, 1),matchedPoints1(i, 2);
              matchedPoints2(i, 1),matchedPoints2(i, 2)];
    viewIDs = [uint32(1), uint32(2)];
    tracks(i) = pointTrack(viewIDs,points);
end

fprintf('\n');
[xyzRefinedPoints,refinedPoses] = ...
    bundleAdjustment(init_xyz_points,tracks,init_poses,cameraParams,...
    'PointsUndistorted',true,'FixedViewIDs',[1,],'Verbose',true,...
    'RelativeTolerance',1e-30, 'AbsoluteTolerance', 1.0,...
    'MaxIterations', 1000);

rotationMatrix = refinedPoses.Orientation{2}';
translationVector = -refinedPoses.Location{2} * rotationMatrix;

scale = norm(translation_vec(1:3)) / norm(translationVector(1:3));
translationVector = translationVector * scale;
xyzRefinedPoints = xyzRefinedPoints * scale;
refined_eul = RotMat2EulAng(rotationMatrix, '213');

fprintf('\nBA Summary\n');
fprintf('GT Rotaion YXZ Angles (degrees): %.17f, %.17f, %.17f\n', gt_eul(1, 1), gt_eul(1, 2), gt_eul(1, 3));
fprintf('Init Rotaion YXZ Angles (degrees): %.17f, %.17f, %.17f\n', init_eul(1, 1), init_eul(1, 2), init_eul(1, 3));
fprintf('Refined Rotaion YXZ Angles (degrees): %.17f, %.17f, %.17f\n', refined_eul(1, 1), refined_eul(1, 2), refined_eul(1, 3));

fprintf('\n');
fprintf('GT Translation (meters): %.17f, %.17f, %.17f\n', translation_vec(1, 1), translation_vec(1, 2), translation_vec(1, 3));
fprintf('Init Translation (meters): %.17f, %.17f, %.17f\n', initTranslationVector(1, 1), initTranslationVector(1, 2), initTranslationVector(1, 3));
fprintf('Refined Translation (meters): %.17f, %.17f, %.17f\n', translationVector(1, 1), translationVector(1, 2), translationVector(1, 3));

% solution 1: ground-truth pose
camMatrix1 = cameraMatrix(cameraParams,eye(3),zeros(1, 3));
camMatrix2 = cameraMatrix(cameraParams,rotation_mat,translation_vec);
camCenter1 = zeros(1, 3);
camCenter2 = -translation_vec * rotation_mat';

[worldPoints,reprojectionErrors] = triangulate(matchedPoints1,matchedPoints2,camMatrix1,camMatrix2);
tri_angles = zeros(size(worldPoints, 1), 1);
for i=1:size(worldPoints, 1)
    point = worldPoints(i, 1:3);
    vec1 = camCenter1 - point;
    vec2 = camCenter2 - point;
    tri_angles(i, 1) = rad2deg(vector_angle(vec1', vec2'));
end

figure;
subplot(151);
scatter3(worldPoints(:,1),worldPoints(:,2),worldPoints(:,3));
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Scene Points');

subplot(152);
tmp = worldPoints - scene_points(:, 1:3);
xy_error = sqrt(sum(tmp(:, 1:2) .* tmp(:, 1:2), 2));
histogram(xy_error);
title('Scene Points XY Errors (Meters)');

subplot(153);
z_error = tmp(:, 3);
histogram(z_error);
title('Scene Points Z Errors (Meters)');

subplot(154);
histogram(reprojectionErrors);
title('Reprojection Errors (Pixels)');

subplot(155);
histogram(tri_angles);
title('Triangulation Angles (degrees)');

sgtitle({'Solution 1: Ground-truth Pose'});


% solution 2: bundle-adjusted poses
camMatrix1 = cameraMatrix(cameraParams,eye(3),zeros(1, 3));
camMatrix2 = cameraMatrix(cameraParams,rotationMatrix,translationVector);
camCenter1 = zeros(1, 3);
camCenter2 = -translationVector * rotationMatrix';

[worldPoints,reprojectionErrors] = triangulate(matchedPoints1,matchedPoints2,camMatrix1,camMatrix2);
tri_angles = zeros(size(worldPoints, 1), 1);
for i=1:size(worldPoints, 1)
    point = worldPoints(i, 1:3);
    vec1 = camCenter1 - point;
    vec2 = camCenter2 - point;
    tri_angles(i, 1) = rad2deg(vector_angle(vec1', vec2'));
end

figure;
subplot(141);
scatter3(worldPoints(:,1),worldPoints(:,2),worldPoints(:,3));
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Scene Points');

subplot(142);
tmp = worldPoints - scene_points(:, 1:3);
tmp = sqrt(sum(tmp .* tmp, 2));
histogram(tmp);
title('Scene Points Errors (Meters)');

subplot(143);
histogram(reprojectionErrors);
title('Reprojection Errors (Pixels)');

subplot(144);
histogram(tri_angles);
title('Triangulation Angles (degrees)');

sgtitle({'Solution 2: Bundle-adjusted Pose'});

figure;
subplot(131);
scale = worldPoints ./ scene_points(:, 1:3);
histogram(scale(:, 1));
title('X axis');

subplot(132);
histogram(scale(:, 2));
title('Y axis');

subplot(133);
histogram(scale(:, 3));
title('Z axis');

sgtitle({'Solution 2 Causes Distortion to the Scene'});

fprintf('\n');
fprintf('Scene Distortion Scale XYZ: %.17f, %.17f, %.17f\n',...
    mean(scale(:,1)), mean(scale(:,2)), mean(scale(:,3)));
fprintf('Sx/Sz-1,Sy/Sz-1: %.17f, %.17f\n', ...
        mean(scale(:,1)./scale(:,3))-1, mean(scale(:,2)./scale(:,3))-1)
fprintf('\n');


function theta = vector_angle(vec1, vec2)
    % inputs are two column vectors
    cos_theta = vec1' * vec2 / sqrt((vec1' * vec1) * (vec2' * vec2));
    theta = acos(cos_theta);
end

% function wrappers for SpinCalc
% assuming [x, y, z] is a row vector, and rotm is used to post-multiply [x,y,z]
% x = 1, y = 2, z = 3
function eul = RotMat2EulAng(rotm, sequence)
%     rotm = rotm';
    eul = SpinCalc(['DCMtoEA', sequence], rotm, 1e-10, 1);
    % make sure the euler angles lie in [-180, 180]
    mask = eul < 180;
    eul = eul .* mask + (eul - 360) .* (1 - mask);
end

function rotm = EulAng2RotMat(eul, sequence)
    rotm = SpinCalc(['EA', sequence, 'toDCM'], eul, 1e-10, 1);
%     rotm = rotm';
end

function rotm = EulVec2RotMat(eul)
    rotm = SpinCalc(['EVtoDCM'], eul, 1e-10, 1);
%     rotm = rotm';
end