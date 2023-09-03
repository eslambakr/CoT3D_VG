# Automatic Location Module 
This module is used to automatically locate the anchors and map each anchor with the most relevant Box ids. 

## How to use

To extract the anchors from the sentences, run the following command:

```
python benchmark_auto_obj_extraction_module_nr3d.py --file_path <path_to_file> --output_path <path_to_output_file>
```

To get the Box_IDs for each anchor, run the following command:

```
python geometry_module.py --file_path <path_to_file> --file_type <file_type> 
```



