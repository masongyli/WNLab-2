# WNLab-2
## Observation 
### Estimate the length of a delimiter
1. A delimiter spans about 90 pixels
2. A `H` or `L` in a delimiter spans about 6 pixels

### Position of double *Da*s
- The 14th, 44th, 74th frame of `0.mp4` contains double *Da*s
- The 25th, 55th, 85th frame of `14.mp4` contains double *Da*s

### Property of autocorrelation
If there is a local minimum, then it should be one of the followings:
- delimiter Da
- delimiter Db
- delimiter Fa
- delimiter Fb
- bit 0
- bit 1

Temporarily call them *tokens*


## TODO
- [x] Find the positions of local minimal autocorrelations in a frame
- [ ] Determine which tokens are there on the positions  (by computing their correlation), and get a sequence of tokens for every frame
- [ ] Use the sequence of tokens in every frame to infer the final bit message
