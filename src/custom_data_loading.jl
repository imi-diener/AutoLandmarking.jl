import Images
"""
    load_imgs(path, dims, numerical)

Reads all the volume images of an entire directory. Data must be stored as follows:
The required input "path" is the path to a directory containing one sub-directory
for every volume that has to be read. Each sub-directory contains the images making
up one volume in .tif format. if the image names contain alphabetical characters
(e.g. image001.tif, image002.tif ect.) the variable "numerical" has to be set to false, if
the image names are just numerical values (e.g 1.tif, 2.tif, 3.tif, ect), it has to
be set to true. Any file not ending in .tif will not be read.

Data will be stored in a 4D array [a x b c x n] with n being the number of
sub-directories, a and b the resolution of the .tif images and c the number of
images. The the resolution of the images has to be the same over all the volumes.
"""
function load_imgs(path, dims, numerical)
    inds = 0
    for (root, dirs, files) in walkdir(path)
        if length(dirs) != 0
            inds = length(dirs)
        end
    end
    image_matrix = zeros(dims[1], dims[2], dims[3], inds)
    vol_names = []
    for (root, dirs, files) in walkdir(path)
        for ind in 1:length(dirs)
          push!(vol_names, dirs[ind])
            ind_path = joinpath(path, dirs[ind])
            items = readdir(ind_path)
            if numerical
                sort!(items, by=x->parse(Int,split(x, ".")[1]))
            end
            for i in items
                if i[end-3:end] != ".tif"
                    deleteat!(items, findall(x->x==i, items))
                end
            end
            for img in 1:length(items)
                im_path = joinpath(ind_path, items[img])
                image = Images.load(im_path)
                image = Images.Gray.(image)
                mat = convert(Array{Float32}, image)
                if size(mat, 1) != 128
                end
                image_matrix[:, :, img, ind] = mat
            end
        end
    end
    return image_matrix, vol_names
end

"""
    read_landmarks(path, num_landmarks, group)

Specific function to read avizo landmark data into an array.
Reads all the files ending in .Ascii in the directory specified
as "path". Data will be stored in a 2D array [c x n] with c being the
number of individual 3D coordinates (30 coordinates in the case of 10 landmarks)
and n being the number of landmark files read.

The variable Group specifies the group of landmarks in the Avizo file that has
to be read (e.g "@1" in the case of group 1)
"""
function read_landmarks(path, num_landmarks, group)
    whole_dir = readdir(path)
    lm_files = []
    for item in whole_dir
        if '.' in item
            if item[end-4:end] == "Ascii"
                push!(lm_files, joinpath(path, item))
            end
        else
            continue
        end
    end
    coordinates = zeros(length(lm_files), num_landmarks * 3)
    for file in 1:length(lm_files)
        f = open(lm_files[file])
        fil = readlines(f)
        close(f)
        coords = []
        counter = 0
        for i in fil
            if i == group
                counter = 1
                continue
            end
            if 1<=counter<=num_landmarks
                push!(coords, i)
                counter+=1
            end
        end
        for point in 1:length(coords)
            single_co = split(coords[point], " ")
            single_co = single_co[1:3]
            for co in 1:3
                if parse(Float64, single_co[co][1:end-5]) * 10.0^parse(Float64, single_co[co][end-1:end]) <= 256
                    coordinates[file, point*3 - (3-co)] = parse(Float64, single_co[co][1:end-5]) * 10.0^parse(Float64, single_co[co][end-1:end])
                else
                    coordinates[file, point*3 - (3-co)] = parse(Float64, single_co[co][1:end-5]) * 10.0^-parse(Float64, single_co[co][end-1:end])
                end
            end
        end
    end
    return convert(Array, coordinates')
end
