# AutoLM.jl Documentation

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
flip_2D(x, y)
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
AutoLM.give_z_value(x)
AutoLM.depth_map(x)
depth_map_all_sides(x)
```

## Handling and utility
```@docs
AutoLM.accuracy(x, y, modell, dimensions)
AutoLM.print_accuracy(x, y, modell, dimensions)
cost_whole_data_3D(x, y)
cost_whole_data_2D(x, y)
AutoLM.avg_accuracy_per_point(modelo, x, y, dims)
AutoLM.predict_single(model, x)
AutoLM.predict_set(X, model)
AutoLM.save_vols_to_folder(folder, vols, names)
AutoLM.array_to_lm_file(output_path, coordinates)
```

## Outlier detection
```@docs
AutoLM.response_distribution(model, X, lms, samples)
AutoLM.to_3d_array(arr)
AutoLM.align_all(arr)
AutoLM.to_2d_array(arr)
AutoLM.mean_shape(arr)
AutoLM.proc_distance(ref, arr)
AutoLM.procrustes_distance_list(arr, names, exclude_highest=false)
AutoLM.align( x :: Matrix{Float64}, y :: Matrix{Float64} )
```
