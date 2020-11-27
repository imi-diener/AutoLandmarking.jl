using LinearAlgebra
using Distances

"""
    response_distribution(model, X, samples)

Takes one sample and returns a distribution of responses for this sample
by setting the model to trainmode! and thus enabling the dropout layers.
"""
function response_distribution(model, X, lms, samples)
  testmode!(model, false)
  out = zeros(size(lms, 1), samples)
  means = []
  st_dev = []
  for i in 1:samples
    out[:,i] = model(X)
  end
  for i in 1:size(out, 1)
    push!(means, mean(out[i, :]))
    push!(st_dev, std(out[i, :]))
  end
  return means, st_dev
end

"""
    to_3d_array(arr)

Takes a 2D landmark array for 3 dimensions as input and outputs a 3D array
with dimensions [n_points x 3 x n_individuals]
"""
function to_3d_array(arr)
  arr_out = zeros(convert(Int32, size(arr, 1)/3),3,size(arr,2))
  counter = 1
  for i in 1:3:size(arr, 1)
    arr_out[counter, 1, :] = arr[i,:]
    arr_out[counter, 2, :] = arr[i+1, :]
    arr_out[counter, 3, :] = arr[i+2, :]
    counter += 1
  end
  return arr_out
end

"""
    align_all(arr)

Performs procrustes alignment on all the landmarks over all the individuals
in a 3D landmark array as returned by the function to_3d_array().
"""
function align_all(arr)
  ref = arr[:,:,1]
  out = deepcopy(arr)
  for i in 2:size(arr,3)
    out[:,:,i] = align(arr[:,:,i], ref)
  end
  return out
end

"""
    to_2d_array(arr)

converts a 3D landmark array to a 2D landmark array with dimensions
[coordinates x individuals] where coordinates are x,y,z coordinates
repeated as many times as there are landmarks.
"""
function to_2d_array(arr)
  n_lms = size(arr, 1)
  out = zeros(n_lms*3, size(arr, 3))
  for i in 1:size(arr, 3)
    for l in 1:n_lms
      out[l*3-2:l*3, i] = arr[l,:,i]
    end
  end
  return out
end

"""
    mean_shape(arr)

finds the mean shape of all teh individuals in a 3D landmark array. Procrustes
alignment should be performed before finding the mean.
"""
function mean_shape(arr)
  n_points = size(arr, 1)
  mean_shape = zeros(n_points, 3)
  for p in 1:n_points
    mean_shape[p, 1] = mean(arr[p, 1, :])
    mean_shape[p, 2] = mean(arr[p, 2, :])
    mean_shape[p, 3] = mean(arr[p, 3, :])
  end
  return mean_shape
end

"""
    proc_distance(ref, arr)

returns the procrustes distances to a reference (e.g. the mean shape)
for every individual in a 3D landmark array.
"""
function proc_distance(ref, arr)
  n_inds = size(arr, 3)
  n_points = size(arr, 1)
  dists = zeros(1, n_inds)
  for i in 1:n_inds
    sum_dist = 0
    for p in 1:n_points
      sum_dist += euclidean(ref[p, :], arr[p, :, i])
    end
    dists[i] = sum_dist
  end
  return dists
end

"""
    procrustes_distance_list(arr, names, exclude_highest=false)

returns the procrustes distances to a reference (e.g. the mean shape)
for every individual in a 3D landmark array.
"""
function procrustes_distance_list(arr, names, exclude_highest=false)
  n_inds = size(arr, 2)
  percentile_list = zeros(1,n_inds)
  arr_3d = to_3d_array(arr)
  proc_aligned = align_all(arr_3d)
  proc_mean = mean_shape(proc_aligned)
  proc_dists = proc_distance(proc_mean, proc_aligned)
  max_dist = maximum(proc_dists)
  min_dist = minimum(proc_dists)
  dists_adj = proc_dists .- min_dist
  dist_range = max_dist - min_dist
  ratings = []
  for i in 1:n_inds
    perc = dists_adj[i] / dist_range
    percentile_list[i] = perc
    if perc<=0.5
      push!(ratings, "***")
    elseif perc<=0.75
      push!(ratings, "**")
    elseif perc<=1.0
      push!(ratings, "*")
    end
  end
  out = hcat(hcat(hcat(names, proc_dists'), percentile_list'), ratings)
  if exclude_highest
    outliers = findall(x->x>=0.9, percentile_list[1,:])
    keepers = setdiff(1:n_inds, outliers)
    proc_mean = mean_shape(proc_aligned[:,:,keepers])
    proc_dists = proc_distance(proc_mean, proc_aligned[:,:,keepers])
    max_dist = maximum(proc_dists)
    min_dist = minimum(proc_dists)
    dists_adj = proc_dists .- min_dist
    dist_range = max_dist - min_dist
    for i in 1:length(keepers)
      perc = dists_adj[i] / dist_range
      out[keepers[i],3] = perc
      if perc<=0.5
        out[keepers[i], 4] = "***"
      elseif perc<=0.75
        out[keepers[i], 4] = "**"
      elseif perc<=1.0
        out[keepers[i], 4] = "*"
      end
    end
    out[outliers, 4] .= "excluded as outlier"
    out[outliers, 3] .= "-"
  end
  show(IOContext(stdout, :limit=>false), MIME"text/plain"(), out)
  return out
end

"""
  function align(x,y)

aligns two structures [sets of points in 3D space]. Solves
the "Procrustes" problem. Structures are expected to be of the same size, and the
correspondence is assumed from the vector indices.

Returns x aligned, by performing the rigid body transformation [rotation
and translation that minimizes the RMSD between x and y].

x, y, and xnew (return) are matrices of dimensions (n,3)
(n is the number of points, 3 is the dimension of the space).

L. Martinez, Institute of Chemistry - University of Campinas
Jan 04, 2019
"""
function align( x :: Matrix{Float64}, y :: Matrix{Float64} )

  n = size(x,1)

  # Computing centroid

  cmx = zeros(3)
  cmy = zeros(3)
  for i in 1:n
    for j in 1:3
      cmx[j] = cmx[j] + x[i,j]
      cmy[j] = cmy[j] + y[i,j]
    end
  end
  cmx = cmx / n
  cmy = cmy / n

  # Translating both sets to the origin

  for i in 1:n
    for j in 1:3
      x[i,j] = x[i,j] - cmx[j]
      y[i,j] = y[i,j] - cmy[j]
    end
  end

  # Computing the quaternion matrix

  xm = Vector{Float64}(undef,n)
  ym = Vector{Float64}(undef,n)
  zm = Vector{Float64}(undef,n)
  xp = Vector{Float64}(undef,n)
  yp = Vector{Float64}(undef,n)
  zp = Vector{Float64}(undef,n)
  for i in 1:n
    xm[i] = y[i,1] - x[i,1]
    ym[i] = y[i,2] - x[i,2]
    zm[i] = y[i,3] - x[i,3]
    xp[i] = y[i,1] + x[i,1]
    yp[i] = y[i,2] + x[i,2]
    zp[i] = y[i,3] + x[i,3]
  end

  q = zeros(4,4)
  for i in 1:n
    q[1,1] = q[1,1] + xm[i]^2 + ym[i]^2 + zm[i]^2
    q[1,2] = q[1,2] + yp[i]*zm[i] - ym[i]*zp[i]
    q[1,3] = q[1,3] + xm[i]*zp[i] - xp[i]*zm[i]
    q[1,4] = q[1,4] + xp[i]*ym[i] - xm[i]*yp[i]
    q[2,2] = q[2,2] + yp[i]^2 + zp[i]^2 + xm[i]^2
    q[2,3] = q[2,3] + xm[i]*ym[i] - xp[i]*yp[i]
    q[2,4] = q[2,4] + xm[i]*zm[i] - xp[i]*zp[i]
    q[3,3] = q[3,3] + xp[i]^2 + zp[i]^2 + ym[i]^2
    q[3,4] = q[3,4] + ym[i]*zm[i] - yp[i]*zp[i]
    q[4,4] = q[4,4] + xp[i]^2 + yp[i]^2 + zm[i]^2
  end
  q[2,1] = q[1,2]
  q[3,1] = q[1,3]
  q[3,2] = q[2,3]
  q[4,1] = q[1,4]
  q[4,2] = q[2,4]
  q[4,3] = q[3,4]

  # Computing the eigenvectors 'v' of the q matrix

  v = LinearAlgebra.eigvecs(q)

  # Compute rotation matrix

  u = Matrix{Float64}(undef,3,3)
  u[1,1] = v[1,1]^2 + v[2,1]^2 - v[3,1]^2 - v[4,1]^2
  u[1,2] = 2. * ( v[2,1]*v[3,1] + v[1,1]*v[4,1] )
  u[1,3] = 2. * ( v[2,1]*v[4,1] - v[1,1]*v[3,1] )
  u[2,1] = 2. * ( v[2,1]*v[3,1] - v[1,1]*v[4,1] )
  u[2,2] = v[1,1]^2 + v[3,1]^2 - v[2,1]^2 - v[4,1]^2
  u[2,3] = 2. * ( v[3,1]*v[4,1] + v[1,1]*v[2,1] )
  u[3,1] = 2. * ( v[2,1]*v[4,1] + v[1,1]*v[3,1] )
  u[3,2] = 2. * ( v[3,1]*v[4,1] - v[1,1]*v[2,1] )
  u[3,3] = v[1,1]^2 + v[4,1]^2 - v[2,1]^2 - v[3,1]^2

  # Rotate vector x [will be stored in xnew], and restore y

  xnew = zeros(n,3)
  for i in 1:n
    for j in 1:3
      for k in 1:3
        xnew[i,j] = xnew[i,j] + u[j,k] * x[i,k]
      end
    end
  end

  # Translate vector to the centroid of y [and restore x and y]

  for i in 1:n
    for j in 1:3
      xnew[i,j] = xnew[i,j] + cmy[j]
      y[i,j] = y[i,j] + cmy[j]
      x[i,j] = x[i,j] + cmx[j]
    end
  end

  return xnew

end
