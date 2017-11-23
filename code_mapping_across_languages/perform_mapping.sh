MAIN="/mount/arbeitsdaten29/corpora/waterloo/img/en_vec/zeroShot/"
CODE="mappingcode/"


EN="hypercos.txt" #<- English Vectors (plain text w2v format) 
DE="de_cow_vecs.txt" #<- Source Language Vectors DE/IT
A="zero_full.align" # Alignment file format word-source TAB word-target (EN)
AV="fullvoc_de.txt" # <- Vocabulary file of the source language (used to predict every word in Source -> Target)
OUT="out/" #<- Output folder

python ${CODE}train_tm.py -o TM1 ${A} ${DE} ${EN}; # Learn the mapping Matrix
python ${CODE}test_tm_pred.py TM1.txt ${AV} ${DE} ${EN};	# Apply the Mapping Matrix
paste -d" " translated_vecs.wds.txt translated_vecs.vecs.txt >> ${OUT}output-vecs-tmp.txt # this is just formating
rm -f translated_vecs*; # remove temporary files
less ${DE} | head -1 > HEAD.txt;
cat HEAD.txt ${OUT}output-vecs-tmp.txt > ${OUT}output-vecs.txt;
rm -f HEAD.txt;
rm -f  ${OUT}output-vecs-tmp.txt
rm -f TM1;
#gzip ${OUT}output-vecs.txt # <- final new file!


# Now we can convert the vectors into binary vectors using the script convert_w2vTXT_to_w2vBIN.py
python convert_w2vTXT_to_w2vBIN.py ${OUT}output-vecs.txt # (will create) output-vecs.txt.bin

# Now we can evaluate the binary embeddings 
# Using hyperscore python AP_evaluation_code/test_norm.py <TaskFile> <Embeddings (Binary)>
# Using default cosine   python AP_evaluation_code/test_default.py <TaskFile> <Embeddings (Binary)>