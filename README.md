# nmsu_cs_chu_sgs

This is the code repository for the paper **Sub-Gibbs Sampling: a New Strategy for Inferring LDA, KDD2017**

1. ```$make``` to compile the project
2. Subsampling with skewed topic distribution property: 
   
   ```$ ./splda nr_topics max_iterations data_dir solver sampler_id```
 
    solver: 0 = Normal LDA
 
    solver: 1 = Sparse LDA
 
    solver: 8 = Alias LDA
 
    solver: 10 = F+LDA - doc-by-doc
 
    solver: 100 = subsampling Normal LDA
 
    solver: 101 = subsampling Sparse LDA
 
    solver: 108 = subsampling Alias LDA
 
    solver: 110 = subsampling F+LDA
 
3. Subsampling with approximate semantics property: 

   ```$ ./setcover_subsampling nr_topics max_iterations data_dir sampler_id solver ratio iratio```

   solver: 200 = subsampling Normal LDA

   solver: 201 = subsampling splda

   solver: 208 = subsampling alias

   solver: 210 = subsampling FLDA-d
