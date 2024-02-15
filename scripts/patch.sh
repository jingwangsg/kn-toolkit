filename=$(readlink -f $1)
new_fn=${filename}_new
appended_rpath=$2

cp $filename $new_fn
chmod +rwx $new_fn
RUNPATH=$(readelf -d $new_fn | grep RUNPATH | grep -oP '\[\K[^]]*')
RPATH=$(readelf -d $new_fn | grep RPATH | grep -oP '\[\K[^]]*')
RPATH=$RUNPATH$RPATH:$appended_rpath
echo "new rpath: $RPATH"
patchelf --remove-rpath $new_fn
patchelf --force-rpath --set-rpath $RPATH $new_fn
echo "patch finish"
readelf -d $new_fn
mv $filename ${filename}.bkp
echo "backup at "${filename}.bkp
mv $new_fn $filename
