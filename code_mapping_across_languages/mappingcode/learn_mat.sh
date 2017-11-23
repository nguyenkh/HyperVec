echo "Training..."


for i in 0 1 2 3 4 5 6 7 8 9
	do
	python ../train_tm.py -o trainmat_${i} align.train.-${i} encow5.ppmi.train-${i} GNet_img_avg.train.-${i} &&
	python ../train_tm.py -o testmat_${i} align.test.-${i} encow5.ppmi.test-${i} GNet_img_avg.test.-${i} 
	done;

