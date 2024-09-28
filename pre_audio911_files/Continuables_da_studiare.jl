using Continuables
# https://docs.juliahub.com/General/Continuables/stable/manual/#Example-of-a-Continuable
# https://github.com/vtjnash/Glob.jl

    skip_folders = ("clips",)

    if !isdir(source_path)
        error("source folder \"$source_path\" doesn't exist.")
    else
        cd(source_path)
        # collect source files
        list_all_juliafiles(path=abspath(".")) = @cont begin
            if isfile(path)
                for i in filename
                    endswith(path, "/" * i) && cont(path)
                end
            elseif isdir(path)
                basename(path) in skip_folders && return
                for file in readdir(path)
                    foreach(cont, list_all_juliafiles(joinpath(path, file)))
                end
            end
        end
        source_files = collect(list_all_juliafiles())
    end