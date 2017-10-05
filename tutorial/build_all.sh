for D in lesson_*/
do
  mkdir -p $D"build"
  cd $D"build"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make
  cd ../../
done

echo ""
echo ""
echo "List of executables:"
ls -1 lesson_*/build/lesson_*
echo ""
echo "To run lesson go to build dir:"
echo "cd lesson_0/build"
echo "./lesson_0"
echo ""
