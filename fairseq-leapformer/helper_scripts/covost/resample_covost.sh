#!/bin/bash

RESAMPLE_RATE=16000

for file in "./fr/clips"/* ; do
	if [ -s "${file}" ] ; then
		new_file="$( echo "${file}" | rev | cut -c 4- | rev )wav"
		sox "${file}" -r ${RESAMPLE_RATE} -G "${new_file}"
	else 
		echo "Empty file...passing"
	fi
done       
