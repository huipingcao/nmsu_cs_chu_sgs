# nmsu_cs_chu_sgs

This is the code repository for the paper **Sub-Gibbs Sampling: a New Strategy for Inferring LDA, ICDM2017**

1. ```$make``` to compile the project
2. Subsampling with skewed topic distribution property: 
   
   ```$ ./splda nr_topics max_iterations data_dir solver sampler_id ratio_q```
 
    solver: 0 = Normal LDA
 
    solver: 1 = Sparse LDA
 
    solver: 8 = Alias LDA
 
    solver: 10 = F+LDA - doc-by-doc
 
    solver: 100 = subsampling Normal LDA
 
    solver: 101 = subsampling Sparse LDA
 
    solver: 108 = subsampling Alias LDA
 
    solver: 110 = subsampling F+LDA
    
    ratio_q is the subsampling ratio for SGS strategy with skewed topic distributions property.
 
3. Subsampling with approximate semantics property: 

   ```$ ./setcover_subsampling nr_topics max_iterations data_dir sampler_id solver ratio_q ratio_r```

   solver: 200 = subsampling Normal LDA

   solver: 201 = subsampling splda

   solver: 208 = subsampling alias

   solver: 210 = subsampling FLDA-d
   
   ratio_q is the subsampling ratio for SGS strategy by utilizing skewed topic distributions property.
   
   ratio_r is the subsampling ratio for SGS strategy by utilizing approximate semantics property.

4. Example shell script files are under *cmd* folder. Remember to change data path before running the script file.

5. Data. We use public data set (NIPS, Enron, NYTimes, PubMed) from UCI reposiroty. Please send us an email for the cleaned data download link.
