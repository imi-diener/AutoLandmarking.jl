# AutoLandmarking.jl Documentation

## Data loading
```@docs
load_imgs(path, dims, numerical)
read_landmarks(path, num_landmarks, group)
```

## Data augmentation
```@docs
flip_volume_front(x, y)
flip_volume_side(x, y)
mirror_vol(x, y)
AutoLandmarking.flip(x, y)
flip_3D(x, y)
jitter_3D(volumes, landmarks, padding)
rotate_images(imgs, lms, deg)
rotate_volumes(vols, lms, deg)
```

## Data preparation
```@docs
swap_xy(lms)
train_test_split_3d(X, y, train_ratio)
regular_train_test_split_3d(X, y)
landmark_to_surface(volumes, landmarks, radius)
image_gradients(x)
align_principal(volumes, landmarks, output_size::Int)
resize_relevant(vols, lms, out_size)
translate_lms_back(lms, reconstruction_array)
choose_dims(y, dims)
AutoLandmarking.give_z_value(x)
AutoLandmarking.depth_map(x)
depth_map_all_sides(x)
```

## Handling and utility
```@docs
AutoLandmarking.accuracy(x, y, modell, dimensions)
AutoLandmarking.print_accuracy(x, y, modell, dimensions)
cost_whole_data_3D(x, y)
cost_whole_data_2D(x, y)
AutoLandmarking.avg_accuracy_per_point(modelo, x, y, dims)
AutoLandmarking.predict_single(model, x)
AutoLandmarking.predict_set(X, model)
AutoLandmarking.save_vols_to_folder(folder, vols, names)
AutoLandmarking.array_to_lm_file(output_path, coordinates)
```

## Outlier detection
```@docs
AutoLandmarking.response_distribution(model, X, lms, samples)
AutoLandmarking.to_3d_array(arr)
AutoLandmarking.align_all(arr)
AutoLandmarking.to_2d_array(arr)
AutoLandmarking.mean_shape(arr)
AutoLandmarking.proc_distance(ref, arr)
AutoLandmarking.procrustes_distance_list(arr, names, exclude_highest=false)
AutoLandmarking.align( x :: Matrix{Float64}, y :: Matrix{Float64} )
```
