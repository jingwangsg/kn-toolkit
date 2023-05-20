filename=$1
appended_rpath=$2

chmod +rwx $filename
RUNPATH=$(readelf -d $filename | grep RUNPATH | grep -oP '\[\K[^]]*')
RPATH=$(readelf -d $filename | grep RPATH | grep -oP '\[\K[^]]*')
RPATH=$RUNPATH$RPATH:$appended_rpath
echo "new rpath: $RPATH"
patchelf --remove-rpath $filename
patchelf --force-rpath --set-rpath $RPATH $filename
echo "patch finish"
readelf -d $filename
