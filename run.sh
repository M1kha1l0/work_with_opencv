if [ "`diff -Z -B main.cpp .temp`" != "" ]; then
    cmake --build build
    cat main.cpp > .temp
fi

ls images > images/.names.txt

./build/opencv-test