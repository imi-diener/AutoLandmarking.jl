import Statistics: mean
using Printf
using DelimitedFiles
import Images
import ImageFiltering
using Base.Threads
using MultivariateStats
using Statistics

"""
    swap_xy(lms)

Swaps x and y coordinates of a landmark array.
"""
function swap_xy(lms)
  new = deepcopy(lms)
  for i in 1:size(lms, 2)
    for cor in 1:3:size(lms,1)
      new[cor, i] = lms[cor+1, i]
      new[cor+1, i] = lms[cor, i]
    end
  end
  return new
end

"""
    empty_3rd_dimension(lms)

returns a new 2D array with 3 coordinates per landmark
with the third (z-) coordinate being 0 for all landmarks.
"""
function empty_3rd_dimension(lms)
    new_lms = zeros(Float32, Int64(size(lms,1)*3/2), size(lms, 2))
    counter = 0
    for i in 1:3:60
        new_lms[i:i+1,:] = lms[i-counter:i-(counter-1), :]
        counter+=1
    end
    return new_lms
end

"""
    change_values!(m, from, to, rel)

Quickly change elements that are `rel`(`from`) to `to`.
"""
function change_values!(m, from, to, rel)
  for index in 1:length(m)
    if rel(m[index], from)
      m[index] = to
    end
  end
end

"""
    reshape_4d_to_5d(tensor)

Takes a 4d tensor with 3d volumes and returns them in a 5d tensor
with one channel per volume.
"""
function reshape_4d_to_5d(tensor)
  img_data = reshape(tensor, (size(tensor)[1],size(tensor)[2],size(tensor)[3], 1, size(tensor)[4]))
  return img_data
end

"""
    resize_2d_images(imgs, lms, out_size)

Resizes 4D tensor of 2D images to size(outsize[1], outsize[2]) and
adjusts landmarks accordingly. out_size is a tuple with sizes in
dim1 and dim2.
"""
function resize_2d_images(imgs, lms, out_size)
    n_inds = size(imgs, 4)
    out = zeros(Float32, out_size[1], out_size[2], size(imgs, 3), n_inds)
    out_lms = deepcopy(lms)
    ratio_x = out_size[1]/size(imgs, 1)
    ratio_y = out_size[2]/size(imgs, 2)
    for i in 1:2:size(lms, 1)
        out_lms[i,:] .= lms[i,:] .* ratio_x
        out_lms[i+1, :] .= lms[i+1, :] .* ratio_y
    end
    for ind in 1:n_inds
        for img in 1:size(imgs, 3)
            out[:,:,img, ind] .= Images.imresize(imgs[:,:,img,ind], out_size)
        end
    end
    return out, out_lms
end

# need to build custom function, the scikit one doesn't work due to dimension problems
"""
    train_test_split_3d(X, y, train_ratio)

Randomly split features and labels with ratio 'train_ratio' and
return training and testing set. Takes a 5d tensor as X.
"""
function train_test_split_3d(X, y, train_ratio)
  indices_train = []
  indices_test = []
  for i in 1:size(X)[5]
    if rand() < train_ratio
      push!(indices_train, i)
    else
      push!(indices_test, i)
    end
  end
  x_train = X[:,:,:,:,indices_train]
  y_train = y[:,indices_train]
  x_test = X[:,:,:,:,indices_test]
  y_test = y[:,indices_test]

  return x_train, x_test, y_train, y_test
end

"""
    regular_train_test_split_3d(X, y)

split features and labels so that every fifth entry in the original
dataset is in the testin set (80/20 split) and
return training and testing set. Takes a 5d tensor as X.
"""
function regular_train_test_split_3d(X, y)
  indices_train = []
  indices_test = []
  for i in 1:size(y)[2]
    if i % 5 != 0
      push!(indices_train, i)
    else
      push!(indices_test, i)
    end
  end
  if length(size(X)) == 5
    x_train = X[:,:,:,:,indices_train]
    y_train = y[:,indices_train]
    x_test = X[:,:,:,:,indices_test]
    y_test = y[:,indices_test]
  else
    x_train = X[:,:,:,indices_train]
    y_train = y[:,indices_train]
    x_test = X[:,:,:,indices_test]
    y_test = y[:,indices_test]
  end
  return x_train, x_test, y_train, y_test
end

"""
    regular_train_test_split_3d(X, y)

split features and labels so that every fifth entry in the original
dataset is in the testin set (80/20 split) and
return training and testing set. Takes a 4d tensor as X.
"""
function regular_train_test_split_2d(X, y)
  indices_train = []
  indices_test = []
  for i in 1:size(X)[4]
    if i % 5 != 0
      push!(indices_train, i)
    else
      push!(indices_test, i)
    end
  end
  x_train = X[:,:,:,indices_train]
  y_train = y[:,indices_train]
  x_test = X[:,:,:,indices_test]
  y_test = y[:,indices_test]

  return x_train, x_test, y_train, y_test
end

"""
    landmark_to_surface(volumes, landmarks, radius)

Moves landmarks that are not already on the volume to the closest
point to them on the surface of the volume.
"""
function landmark_to_surface(volumes, landmarks, radius)
  inds = size(landmarks, 2)
  lm_out = deepcopy(landmarks)
  for ind in 1:inds
    image = volumes[:,:,:,ind]
    maxi = maximum(volumes[:,:,:,ind])
    landm = landmarks[:, ind]
    n_points = floor(Int, length(landm)/3)
    coordinates_lm = zeros(n_points, 3)
    for i in 1:n_points
        for j in 1:3
            coordinates_lm[i,j] = landm[3*i-(3-j)]
        end
    end
    lm_coord = floor.(Int, round.(coordinates_lm .*10))
    for point in 1:size(lm_coord, 1)
      if image[Base.max(1, lm_coord[point, 1]), Base.max(1, lm_coord[point, 2]), Base.max(1, lm_coord[point, 3])] != maxi
        continue
      end
      points_in_area = findall(x->x!=maxi, image[Base.max(1, lm_coord[point, 1]-radius):Base.min(size(image, 1), lm_coord[point, 1]+radius),
      Base.max(1, lm_coord[point, 2]-radius):Base.min(size(image, 2), lm_coord[point, 2]+radius),
      Base.max(1, lm_coord[point, 3]-radius):Base.min(size(image, 3), lm_coord[point, 3]+radius)])
      if length(points_in_area) == 0
        continue
      end
      nearest = [0, 10000000]
      for p in 1:length(points_in_area)
        dist = sum([(radius-points_in_area[p][1])^2, (radius-points_in_area[p][2])^2,
        (radius-points_in_area[p][3])^2])
        if dist < nearest[2]
          nearest[2] = dist
          nearest[1] = p
        end
      end
      lm_out[point*3-2, ind] = (points_in_area[nearest[1]][1] + lm_coord[point, 1]-radius)/10
      lm_out[point*3-1, ind] = (points_in_area[nearest[1]][2] + lm_coord[point, 2]-radius)/10
      lm_out[point*3, ind] = (points_in_area[nearest[1]][3] + lm_coord[point, 3]-radius)/10
    end
  end
  return lm_out
end

"""
    image_gradients(x)

Adds the sum of the image gradients in x and y direction for each channel
of each image in a 4-dimensional tensor as a new channel.
"""
function image_gradients(x)
  copyx = deepcopy(x)
  for ind in 1:size(x, 4)
    for chan in 1:size(x, 3)
      gradix, gradiy = ImageFiltering.imgradients(x[:,:,chan,ind], ImageFiltering.Kernel.ando3)
      copyx[:,:,chan,ind] = abs.(gradix) .+ abs.(gradiy)
    end
  end
  out = cat(x, copyx, dims=3)
  return out
end

"""
    align_principal(volumes, landmarks, output_size)

Aligns all volumes in tensor "volumes" along their principal
axis (new z-axis) and transforms respective landmark data alongside.
The output size can be specified with output_size
(a cube with size output_size^3 will be returned), but must be
large enough to contain the largest aligned volume. a median filter
will be applied to handle caused by transformation. Also returns
a reconstruction array used to translate predicted landmarks back
to their original volume.
"""
function align_principal(volumes, landmarks, output_size::Int)
  inds = size(landmarks, 2)
  out = []
  lm_out = deepcopy(landmarks)
  reconstruction = zeros(3, 5, inds)
  for ind in 1:inds
    image = volumes[:,:,:,ind]
    maxi = maximum(volumes[:,:,:,ind])
    points = findall(x->x!=maxi, image)
    coordinates = zeros(length(points), 3)
    for i in 1:length(points)
        for j in 1:3
            coordinates[i,j] = points[i][j]
        end
    end
    pcs = fit(PCA, coordinates', maxoutdim=5)
    tra = transform(pcs, coordinates')
    positive = tra .- (minimum(tra)-1)
    maxi = round(maximum(positive))
    pos = floor.(Int, round.(positive))
    vol2 = zeros(output_size, output_size, output_size, 1)
    for pnt in 1:size(pos,2)
        vol2[pos[1, pnt], pos[2, pnt], pos[3, pnt], 1] = 1.0
    end
    vol3 = deepcopy(vol2)
    # for i in 1:output_size
    #   vol3[:,:,i,1] = ImageFiltering.mapwindow(median!, vol2[:,:,i,1], (3,3))
    # end
    change_values!(vol3, 0.0, 2.0, ==)
    if ind == 1
      out = vol3
    else
      out = cat(out, vol3, dims = 4)
    end
    landm = landmarks[:, ind]
    n_points = floor(Int, length(landm)/3)
    coordinates_lm = zeros(n_points, 3)
    for i in 1:n_points
        for j in 1:3
            coordinates_lm[i,j] = landm[3*i-(3-j)]
        end
    end
    lm_coord = coordinates_lm .*10
    new_lm = transform(pcs, lm_coord')
    pos_lm = new_lm .- (minimum(tra)-1)
    for i in 1:n_points
        for j in 1:3
            lm_out[3*i-(3-j), ind] = pos_lm[j, i]
        end
    end
    reconstruction[:,1:3, ind] = projection(pcs)
    reconstruction[1,4, ind] = minimum(tra)
    reconstruction[2,4, ind] = 1
    reconstruction[:,5,ind] = mean(pcs)
  end
  return out, relu.(lm_out ./ 10), reconstruction
end

"""
    resize_relevant(vols, lms, out_size)

Takes the output from align_principal and resizes the relevant
part (part containing the actual item) to a cube of specified size.
Also returns the ratio between resized and original volume size.
Adjusts landmarks.
"""
function resize_relevant(vols, lms, out_size)
  inds = size(vols, 4)
  out = zeros(out_size, out_size, out_size, inds)
  scales = []
  lms_out = deepcopy(lms)
  for ind in 1:inds
    maxi = maximum(vols[:,:,:,ind])
    coords = findall(x->x!=maxi, vols[:,:,:,ind])
    max_coord = maximum(coords)
    ar = zeros(1,3)
    for i in 1:3
      ar[i] = max_coord[i]
    end
    maximu = Int(maximum(ar))
    scale = out_size/maximu
    push!(scales, scale)
    out[:,:,:,ind] = Images.imresize(vols[1:maximu,1:maximu,1:maximu,ind], (out_size, out_size, out_size))
    lms_out[:, ind] = scale .* lms[:, ind]
  end
  return out, lms_out, scales
end

"""
    translate_lms_back(lms, reconstruction_array)

Takes an array of predicted landmarks and a reconstruction array
as returned by align_principal() and translates the landmarks
back to their respective original volume.
"""
function translate_lms_back(lms, reconstruction_array)
  lm_out = deepcopy(lms)
  for ind in 1:size(lms, 2)
    landm = lms[:,ind]
    n_points = floor(Int, length(landm)/3)
    coordinates_lm = zeros(n_points, 3)
    for i in 1:n_points
        for j in 1:3
            coordinates_lm[i,j] = landm[3*i-(3-j)]
        end
    end
    coord = coordinates_lm .* 10
    coord_scaled = coord ./ reconstruction_array[2,4,ind]
    coord_center = coord_scaled .+ (reconstruction_array[1,4,ind] - 1)
    new_coord = (reconstruction_array[1:3,1:3,ind] * coord_center')

    for i in 1:n_points
        for j in 1:3
            lm_out[3*i-(3-j), ind] = new_coord[j, i] + reconstruction_array[j, 5, ind]
        end
    end
  end
  return lm_out ./ 10
end

"""
    choose_dims(y, dims)

choose which dimensions of x,y and z should be included in the labels and return
a new array containing only the specified dims.
"""
function choose_dims(y, dims)
  newy = []
  for i in 1:3:size(y)[1]
    for o in 1:3
      if dims[o] == 1
        if size(newy)[1] == 0
          newy = y[i+(o-1), :]
        else
          newy = hcat(newy, y[i+(o-1), :])
        end
      else
        continue
      end
    end
  end
  return newy'
end

"""
    give_z_value(x)

sets every value outside of the actual volume to 0 and changes the voxel-values
of every voxel within the volume to its corresponding z-coordinate. Returns
the new tensor with all values normalized between -1 and 1.
"""
function give_z_value(x)
  copyx = deepcopy(x)
  Threads.@threads for ind in 1:size(x)[4]
    siz = size(x)[3]
    for i in 1:siz
      img = x[:,:,i, ind]
      ma = maximum(img)
      img[findall(x->x==ma, img)] .= -1
      img[findall(x->x!=-1, img)] .= (2*((i-1)/(siz-1))-1)*-1
      copyx[:,:,i,ind] = img
    end
  end
  return copyx
end

"""
    make_elevation_map(x)

Returns a 2d image for every volume in tensor x and sets the pixel values
to the maximum value within each column of voxels in dimension z as returned
by the function give_z_value().
"""
function depth_map(x)
  max_val = maximum(x)
  x_out = zeros(Float32, size(x)[1], size(x)[2], 1, size(x)[4])
  Threads.@threads for inds in 1:size(x)[4]
    for dim1 in 1:size(x)[1]
      for dim2 in 1:size(x)[2]
        x_out[dim1, dim2, 1, inds] = maximum(x[dim1,dim2,:,inds])/max_val
      end
    end
  end
  return x_out
end

"""
    depth_map_all_sides(x)

returns a 4d image tensor with 6 channels per image, each representing
a depthmap from one side of the volume.
"""
function depth_map_all_sides(x)
  mock_y = zeros(3, size(x, 4))
  first = give_z_value(x)
  second = -1 .* first
  change_values!(second, 1.0, -1.0, ==)
  out = depth_map(first)
  out = cat(out, depth_map(second), dims=3)
  first = nothing
  second = nothing
  front_back, y_ = flip_volume_front(x, mock_y)
  third = give_z_value(front_back)
  forth = -1 .* third
  change_values!(forth, 1.0, -1.0, ==)
  out = cat(out, depth_map(third), dims=3)
  out = cat(out, depth_map(forth), dims=3)
  third = nothing
  forth = nothing
  side_side, y_ = flip_volume_side(front_back, mock_y)
  front_back = nothing
  fifth = give_z_value(side_side)
  sixth = -1 .* fifth
  change_values!(sixth, 1.0, -1.0, ==)
  out = cat(out, depth_map(fifth), dims=3)
  out = cat(out, depth_map(sixth), dims=3)
  return out
end
