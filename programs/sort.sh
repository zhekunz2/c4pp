for f in `find .`; do mv -v "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done
for entry in ./bugs_examples/vol1/*
	do
		#subfolder_name = $(basename "$entry")
		echo $(basename "$entry")
		mkdir "$entry"/stan_code
		for file in "$entry"/*
			do
				#echo "$file"
				mv "$file" "$entry"/stan_code
			done
		mkdir "$entry"/bug_code
		python generate_script.py "$(basename "$entry")" >> "$entry"/bug_code/script.txt
		find ./examples/ -iname "$(basename "$entry")model.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")data.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")inits.txt" -exec cp  {} "$entry"/bug_code \;
	done	
for entry in ./bugs_examples/vol2/*
	do
		echo $(basename "$entry")
		mkdir "$entry"/stan_code
		for file in "$entry"/*
			do
				mv "$file" "$entry"/stan_code
			done
		mkdir "$entry"/bug_code
		python generate_script.py "$(basename "$entry")" >> "$entry"/bug_code/script.txt
		find ./examples/ -iname "$(basename "$entry")model.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")data.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")inits.txt" -exec cp  {} "$entry"/bug_code \;
	done
for entry in ./bugs_examples/vol3/*
	do
		echo $(basename "$entry")
		mkdir "$entry"/stan_code
		for file in "$entry"/*
			do
				mv "$file" "$entry"/stan_code
			done
		mkdir "$entry"/bug_code
		python generate_script.py "$(basename "$entry")" >> "$entry"/bug_code/script.txt
		find ./examples/ -iname "$(basename "$entry")model.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")data.txt" -exec cp  {} "$entry"/bug_code \;
		find ./examples/ -iname "$(basename "$entry")inits.txt" -exec cp  {} "$entry"/bug_code \;
	done
