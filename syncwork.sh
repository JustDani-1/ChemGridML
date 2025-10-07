#!/bin/bash

git pull
rm -rf ./master.log

####################
# Move output files
####################

mkdir -p output
# Find and process all files matching the pattern
for file in *.[eo]*; do
    # Skip if no matching files found (when glob doesn't expand)
    [[ ! -e "$file" ]] && continue
    
    # Extract JobID using regex
    if [[ "$file" =~ ^(.+)\.[eo]([0-9]+)(\.[0-9]+)?$ ]]; then

        job_id="${BASH_REMATCH[2]}"
        
        # Create subdirectory for this JobID
        target_dir="./output/${job_id}"
        mkdir -p "$target_dir"
        
        # Move the file
        echo "Moving $file to $target_dir/"
        mv "$file" "$target_dir/"
    else
        echo "Warning: File '$file' doesn't match expected pattern"
    fi
done

####################
# remove .db clutter
####################

rm -rf ./studies/*/studies/*.db-wal
rm -rf ./studies/*/studies/*.db-shm

####################
# push the results
####################

git add *
git commit -m 'Myriad done'
git push