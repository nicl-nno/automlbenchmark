echo "Start FEDOT benchmarking"

## Update FEDOT ##
./frameworks/FEDOT/setup.sh

## Run benchmarks ##
python3 runbenchmark.py FEDOT small

## Get result ##
cd results
latest_results=$(ls -td -- */ | grep fedot | head -n 1)
echo $latest_results
cp $latest_results ~/nfs/40/share_0/fedot-daily-benchmarks/$latest_results -r
cd ..

echo Finished



