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
AutoLM.flip_2D(x, y)
flip_3D(x, y)
jitter_3D(volumes, landmarks, padding)
rotate_images(imgs, lms, deg)
rotate_volumes(vols, lms, deg)
```
