SET="generated"
dir="/esat/qayd/kkelchte/simulation/$SET"
rm list_of_lengths.txt

for f in $dir/*
do
    if [[ $f != "$dir/cache" &&  $f != "$dir/test_set.txt"&&  $f != "$dir/train_set.txt"&&  $f != "$dir/val_set.txt" ]];
    then
        #check num of images
        rgbnum="$(ls $f/RGB | wc -l)"
        #add name f to list of that num of images
        #file="tmp/$rgbnum.txt"
        #echo $f >> $file
        message="$rgbnum - $f"
        echo $message >> list_of_lengths.txt
        
    fi
done

for f in tmp/*
do
    num="$(cat $f | wc -l)"
    #echo $num
done
