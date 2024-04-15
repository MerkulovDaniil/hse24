for file in "$@"
do
    dir=$(dirname "$file")  # Get the directory path of the original file
    base=$(basename "$file" .pdf)  # Get the base name of the file
    # Move and rename files from Desktop to the original file's directory
    counter=0
    # Correct the pattern to match generated files from 'Extract PDF Pages'
    shopt -s nullglob  # Ensure pattern matching does not return unmatched patterns
    echo "$dir"
    echo "$base"
    for page in ~/Desktop/"$base"-page*.pdf
    do
        newname="$dir/${base}$counter.pdf"
        mv "$page" "$newname"
        let counter++
    done
done
