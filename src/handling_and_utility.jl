import Statistics: mean
using Printf
using DelimitedFiles
using Base.Threads
using MultivariateStats
using Statistics
using Flux
using CuArrays
"""
    accuracy(x, y, modell)

calculate euclidean distances between points in y and y' for all samples
"""
function accuracy(x, y, modell, dimensions)
  testmode!(modell)
  resp = modell(x)
  testmode!(modell, false)
  resp = relu.(resp)
  mean_accs = zeros(convert(Int32,size(y,1)/dimensions), size(y,2))
  counter = 1
  for i in 1:dimensions:size(y)[1]
    coordsy = y[i:i+(dimensions-1), :]
    coordsresp = resp[i:i+(dimensions-1), :]
    dist = (coordsy .- coordsresp).^2
    dists = zeros(size(dist,2), 1)
    for j in 1:size(dist)[2]
      dists[j, 1] = sqrt(sum(dist[:,j]))
    end
    mean_accs[counter, :] = dists'[:,1]
    counter += 1
  end
  return mean_accs
end

"""
    print_accuracy(x, y, modell)

calculate and print euclidean distances between points in y and y' for all samples
"""
function print_accuracy(x, y, modell, dimensions)
  testmode!(modell)
  resp = modell(x)
  testmode!(modell, false)
  mean_accs = []
  counter = 1
  for i in 1:dimensions:size(y)[1]
    coordsy = y[i:i+(dimensions-1), :]
    coordsresp = resp[i:i+(dimensions-1), :]
    dist = (coordsy .- coordsresp).^2
    dists = []
    for j in 1:size(dist)[2]
      push!(dists, sqrt(sum(dist[:,j])))
    end
    println("mean distance on point ", counter, " is ", mean(dists))
    push!(mean_accs, dists)
    counter += 1
  end
  return mean_accs
end

"""
    cost_whole_data(x, y)

calculates cost (as defined by function 'cost') over an entire 5d tensor
containing 3d image data.
"""
function cost_whole_data_3D(x, y)
  costs = []
  for i in 1:size(y)[2]
    push!(costs, cost(x[:,:,:,:,i:i], y[:,i:i]))
  end
  return sum(costs)
end

"""
    cost_whole_data_2D(x, y)

calculates cost (as defined by function 'cost') over an entire 4d tensor
containing 2d image data.
"""
function cost_whole_data_2D(x, y)
  costs = []
  for i in 1:4:size(y)[2]
    if i+3>size(y, 2)
      push!(costs, cost(x[:,:,:,i:end], y[:,i:end]))
    else
      push!(costs, cost(x[:,:,:,i:i+3], y[:,i:i+3]))
    end
  end
  return sum(costs)
end

"""
    avg_accuracy_per_point(modelo, x, y, dims)

calculates average, min and max distance between predicted and actual landmarks.
"""
function avg_accuracy_per_point(modelo, x, y, dims)
  devs = []
  for i in 1:size(y)[2]
    if size(devs)[1] == 0
      devs = accuracy(x[:,:,:,i:i], y[:,i:i], modelo, dims)
    else
      devs = hcat(devs, accuracy(x[:,:,:,i:i], y[:,i:i], modelo, dims))
    end
  end
  max_acc = []
  min_acc = []
  avg_accs = []
  for i in 1:size(devs)[1]
    push!(avg_accs, median(devs[i, :]))
    push!(min_acc, minimum(devs[i, :]))
    push!(max_acc, maximum(devs[i, :]))
  end
  return avg_accs, max_acc, min_acc
end

"""
    predict_single(model, x)

Returns the output of 'model' in testing mode given input x and applies the relu
function to each output. Takes a single sample.
"""
function predict_single(model, x)
  testmode!(model)
  resp = model(x)
  resp = relu.(resp)
  testmode!(model, false)
  return resp
end

"""
    predict_set(X, model)

Returns the output of 'model' in testing mode given input x and applies the relu
function to each output. Takes multiple samples.
"""
function predict_set(X, model)
  resps = []
  testmode!(model)
  for i in 1:size(X, 4)
    if i == 1
      resps = model(X[:,:,:,i:i])
    else
      resps = cat(resps, model(X[:,:,:,i:i]), dims=2)
    end
  end
  testmode!(model, false)
  return relu.(resps)
end


"""
    save_vols_to_folder(folder, vols, names)

Save a 4D tensor containing 3D volumes to a folder, each in its own
sub folder,  in .tif format.
"""
function save_vols_to_folder(folder, vols, names)
  for i in 1:length(names)
    dir = joinpath(folder, names[i])
    mkdir(dir)
    for img in 1:size(vols, 3)
      name = joinpath(dir, string(img, ".tif"))
      Images.save(name, Images.colorview(Images.Gray, vols[:,:,img,i]))
    end
  end
end

"""
    array_to_lm_file(output_path, coordinates)


Create an avizo landmark file (output_path) with one
set of landmarks. The coordinates to these landmarks need to be passed
in the form of a one-dimensional array as returned by a network when one
volume is given as the input.
"""
function array_to_lm_file(output_path, coordinates)
  io = open(output_path, "w")
  num_lms = 0
  temp = []
  for i in 1:3:length(coordinates)
    push!(temp, string(@sprintf("%.15e ", coordinates[i]),@sprintf("%.15e ", coordinates[i+1]),
    @sprintf("%.15e ", coordinates[i+2])))
    num_lms += 1
  end
  write(io, string("""# Avizo 3D ASCII 3.0


  define Markers $(num_lms)

  Parameters {
  \tNumSets 1,
  \tContentType "LandmarkSet"
  }

  Markers { float[3] Coordinates } @1

  # Data section follows
  @1
  """))
  close(io)
  io = open(output_path, "a")
  writedlm(io, temp,'\n')
  close(io)
end
