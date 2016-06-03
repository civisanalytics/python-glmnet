c
c                          newGLMnet (5/12/14)
c
c
c                 Elastic net with squared-error loss
c
c dense predictor matrix:
c
c call elnet(ka,parm,no,ni,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,isd,
c            intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
c
c   x(no,ni) = predictor data matrix flat file (overwritten)
c
c
c sparse predictor matrix:
c
c call spelnet(ka,parm,no,ni,x,ix,jx,y,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
c             isd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
c
c   x, ix, jx = predictor data matrix in compressed sparse row format
c
c
c other inputs:
c
c   ka = algorithm flag
c      ka=1 => covariance updating algorithm
c      ka=2 => naive algorithm
c   parm = penalty member index (0 <= parm <= 1)
c        = 0.0 => ridge
c        = 1.0 => lasso
c   no = number of observations
c   ni = number of predictor variables
c   y(no) = response vector (overwritten)
c   w(no)= observation weights (overwritten)
c   jd(jd(1)+1) = predictor variable deletion flag
c      jd(1) = 0  => use all variables
c      jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
c   vp(ni) = relative penalties for each predictor variable
c      vp(j) = 0 => jth variable unpenalized
c   cl(2,ni) = interval constraints on coefficient values (overwritten)
c      cl(1,j) = lower bound for jth coefficient value (<= 0.0)
c      cl(2,j) = upper bound for jth coefficient value (>= 0.0)
c   ne = maximum number of variables allowed to enter largest model
c        (stopping criterion)
c   nx = maximum number of variables allowed to enter all models
c        along path (memory allocation, nx > ne).
c   nlam = (maximum) number of lamda values
c   flmin = user control of lamda values (>=0)
c      flmin < 1.0 => minimum lamda = flmin*(largest lamda value)
c      flmin >= 1.0 => use supplied lamda values (see below)
c   ulam(nlam) = user supplied lamda values (ignored if flmin < 1.0)
c   thr = convergence threshold for each lamda solution.
c      iterations stop when the maximum reduction in the criterion value
c      as a result of each parameter update over a single pass
c      is less than thr times the null criterion value.
c      (suggested value, thr=1.0e-5)
c   isd = predictor variable standarization flag:
c      isd = 0 => regression on original predictor variables
c      isd = 1 => regression on standardized predictor variables
c      Note: output solutions always reference original
c            variables locations and scales.
c   intr = intercept flag
c      intr = 0/1 => don't/do include intercept in model
c   maxit = maximum allowed number of passes over the data for all lambda
c      values (suggested values, maxit = 100000)
c
c output:
c
c   lmu = actual number of lamda values (solutions)
c   a0(lmu) = intercept values for each solution
c   ca(nx,lmu) = compressed coefficient values for each solution
c   ia(nx) = pointers to compressed coefficients
c   nin(lmu) = number of compressed coefficients for each solution
c   rsq(lmu) = R**2 values for each solution
c   alm(lmu) = lamda values corresponding to each solution
c   nlp = actual number of passes over the data for all lamda values
c   jerr = error flag:
c      jerr  = 0 => no error
c      jerr > 0 => fatal error - no output returned
c         jerr < 7777 => memory allocation error
c         jerr = 7777 => all used predictors have zero variance
c         jerr = 10000 => maxval(vp) <= 0.0
C      jerr < 0 => non fatal error - partial output:
c         Solutions for larger lamdas (1:(k-1)) returned.
c         jerr = -k => convergence for kth lamda value not reached
c            after maxit (see above) iterations.
c         jerr = -10000-k => number of non zero coefficients along path
c            exceeds nx (see above) at kth lamda value.
c
c
c
c least-squares utility routines:
c
c
c uncompress coefficient vectors for all solutions:
c
c call solns(ni,nx,lmu,ca,ia,nin,b)
c
c input:
c
c    ni,nx = input to elnet
c    lmu,ca,ia,nin = output from elnet
c
c output:
c
c    b(ni,lmu) = all elnet returned solutions in uncompressed format
c
c
c uncompress coefficient vector for particular solution:
c
c call uncomp(ni,ca,ia,nin,a)
c
c input:
c
c    ni = total number of predictor variables
c    ca(nx) = compressed coefficient values for the solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for the solution
c
c output:
c
c    a(ni) =  uncompressed coefficient vector
c             referencing original variables
c
c
c evaluate linear model from compressed coefficients and
c uncompressed predictor matrix:
c
c call modval(a0,ca,ia,nin,n,x,f);
c
c input:
c
c    a0 = intercept
c    ca(nx) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    n = number of predictor vectors (observations)
c    x(n,ni) = full (uncompressed) predictor matrix
c
c output:
c
c    f(n) = model predictions
c
c
c evaluate linear model from compressed coefficients and
c compressed predictor matrix:
c
c call cmodval(a0,ca,ia,nin,x,ix,jx,n,f);
c
c input:
c
c    a0 = intercept
c    ca(nx) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    x, ix, jx = predictor matrix in compressed sparse row format
c    n = number of predictor vectors (observations)
c
c output:
c
c    f(n) = model predictions
c
c
c
c
c                           Multiple response
c                  elastic net with squared-error loss
c
c dense predictor matrix:
c
c call multelnet(parm,no,ni,nr,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,isd,
c                jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
c
c   x(no,ni) = predictor data matrix flat file (overwritten)
c
c
c sparse predictor matrix:
c
c call multspelnet(parm,no,ni,nr,x,ix,jx,y,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
c             isd,jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
c
c   x, ix, jx = predictor data matrix in compressed sparse row format
c
c other inputs:
c
c   nr = number of response variables
c   y(no,nr) = response data matrix (overwritten)
c   jsd = response variable standardization flag
c      jsd = 0 => regression using original response variables
c      jsd = 1 => regression using standardized response variables
c      Note: output solutions always reference original
c            variables locations and scales.
c   all other inputs same as elnet/spelnet above
c
c output:
c
c   a0(nr,lmu) = intercept values for each solution
c   ca(nx,nr,lmu) = compressed coefficient values for each solution
c   all other outputs same as elnet/spelnet above
c   (jerr = 90000 => bounds adjustment non convergence)
c
c
c
c multiple response least-squares utility routines:
c
c
c uncompress coefficient matrix for all solutions:
c
c call multsolns(ni,nx,nr,lmu,ca,ia,nin,b)
c
c input:
c
c    ni,nx,nr = input to multelnet
c    lmu,ca,ia,nin = output from multelnet
c
c output:
c
c    b(ni,nr,lmu) = all multelnet returned solutions in uncompressed format
c
c
c uncompress coefficient matrix for particular solution:
c
c call multuncomp(ni,nr,nx,ca,ia,nin,a)
c
c input:
c
c    ni,nr,nx = input to multelnet
c    ca(nx,nr) = compressed coefficient values for the solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for the solution
c
c output:
c
c    a(ni,nr) =  uncompressed coefficient matrix
c             referencing original variables
c
c
c evaluate linear model from compressed coefficients and
c uncompressed predictor matrix:
c
c call multmodval(nx,nr,a0,ca,ia,nin,n,x,f);
c
c input:
c
c    nx,nr = input to multelnet
c    a0(nr) = intercepts
c    ca(nx,nr) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    n = number of predictor vectors (observations)
c    x(n,ni) = full (uncompressed) predictor matrix
c
c output:
c
c    f(nr,n) = model predictions
c
c
c evaluate linear model from compressed coefficients and
c compressed predictor matrix:
c
c call multcmodval(nx,nr,a0,ca,ia,nin,x,ix,jx,n,f);
c
c input:
c
c    nx,nr = input to multelnet
c    a0(nr) = intercepts
c    ca(nx,nr) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    x, ix, jx = predictor matrix in compressed sparse row format
c    n = number of predictor vectors (observations)
c
c output:
c
c    f(nr,n) = model predictions
c
c
c
c
c          Symmetric binomial/multinomial logistic elastic net
c
c
c dense predictor matrix:
c
c call lognet (parm,no,ni,nc,x,y,o,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,isd,
c              intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
c
c   x(no,ni) = predictor data matrix flat file (overwritten)
c
c
c sparse predictor matrix:
c
c call splognet (parm,no,ni,nc,x,ix,jx,y,o,jd,vp,cl,ne,nx,nlam,flmin,
c      ulam,thr,isd,intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
c
c   x, ix, jx = predictor data matrix in compressed sparse row format
c
c
c other inputs:
c
c   parm,no,ni,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,isd,intr,maxit
c    = same as elnet above.
c
c   nc = number of classes (distinct outcome values)
c        nc=1 => binomial two-class logistic regression
c            (all output references class 1)
c   y(no,max(2,nc)) = number of each class at each design point
c      entries may have fractional values or all be zero (overwritten)
c   o(no,nc) = observation off-sets for each class
c   kopt = optimization flag
c      kopt = 0 => Newton-Raphson (recommended)
c      kpot = 1 => modified Newton-Raphson (sometimes faster)
c      kpot = 2 => nonzero coefficients same for each class (nc > 1)
c
c
c output:
c
c   lmu,ia,nin,alm,nlp = same as elent above
c
c   a0(nc,lmu) = intercept values for each class at each solution
c   ca(nx,nc,lmu) = compressed coefficient values for each class at
c                each solution
c   dev0 = null deviance (intercept only model)
c   fdev(lmu) = fraction of devience explained by each solution
c   jerr = error flag
c      jerr = 0  => no error
c      jerr > 0 => fatal error - no output returned
c         jerr < 7777 => memory allocation error
c         jerr = 7777 => all used predictors have zero variance
c         jerr = 8000 + k => null probability < 1.0e-5 for class k
c         jerr = 9000 + k => null probability for class k
c                            > 1.0 - 1.0e-5
c         jerr = 10000 => maxval(vp) <= 0.0
c         jerr = 90000 => bounds adjustment non convergence
C      jerr < 0 => non fatal error - partial output:
c         Solutions for larger lamdas (1:(k-1)) returned.
c         jerr = -k => convergence for kth lamda value not reached
c            after maxit (see above) iterations.
c         jerr = -10000-k => number of non zero coefficients along path
c            exceeds nx (see above) at kth lamda value.
c         jerr = -20000-k => max(p*(1-p)) < 1.0e-6 at kth lamda value.
c    o(no,nc) = training data values for last (lmu_th) solution linear
c               combination.
c
c
c
c logistic/multinomial utilitity routines:
c
c
c uncompress coefficient vectors for all solutions:
c
c call lsolns(ni,nx,nc,lmu,ca,ia,nin,b)
c
c input:
c
c    ni,nx,nc = input to lognet
c    lmu,ca,ia,nin = output from lognet
c
c output:
c
c    b(ni,nc,lmu) = all lognet returned solutions in uncompressed format
c
c
c uncompress coefficient vector for particular solution:
c
c call luncomp(ni,nx,nc,ca,ia,nin,a)
c
c input:
c
c    ni, nx, nc = same as above
c    ca(nx,nc) = compressed coefficient values (for each class)
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients
c
c output:
c
c    a(ni,nc) =  uncompressed coefficient vectors
c                 referencing original variables
c
c
c evaluate linear model from compressed coefficients and
c uncompressed predictor vectors:
c
c call lmodval(nt,x,nc,nx,a0,ca,ia,nin,ans);
c
c input:
c
c    nt = number of observations
c    x(nt,ni) = full (uncompressed) predictor vectors
c    nc, nx = same as above
c    a0(nc) = intercepts
c    ca(nx,nc) = compressed coefficient values (for each class)
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients
c
c output:
c
c ans(nc,nt) = model predictions
c
c
c evaluate linear model from compressed coefficients and
c compressed predictor matrix:
c
c call lcmodval(nc,nx,a0,ca,ia,nin,x,ix,jx,n,f);
c
c input:
c
c    nc, nx = same as above
c    a0(nc) = intercept
c    ca(nx,nc) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    x, ix, jx = predictor matrix in compressed sparse row format
c    n = number of predictor vectors (observations)
c
c output:
c
c    f(nc,n) = model predictions
c
c
c
c
c                        Poisson elastic net
c
c
c dense predictor matrix:
c
c call fishnet (parm,no,ni,x,y,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
c               isd,intr,maxit,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
c
c   x(no,ni) = predictor data matrix flat file (overwritten)
c
c sparse predictor matrix:
c
c call spfishnet (parm,no,ni,x,ix,jx,y,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
c               isd,intr,maxit,lmu,a0,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
c
c    x, ix, jx = predictor data matrix in compressed sparse row format
c
c other inputs:
c
c   y(no) = observation response counts
c   o(no) = observation off-sets
c   parm,no,ni,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,isd,intr,maxit
c    = same as elnet above
c
c output:
c
c   lmu,a0,ca,ia,nin,alm = same as elnet above
c   dev0,fdev = same as lognet above
c   nlp = total number of passes over predictor variables
c   jerr = error flag
c      jerr = 0  => no error
c      jerr > 0 => fatal error - no output returned
c         jerr < 7777 => memory allocation error
c         jerr = 7777 => all used predictors have zero variance
c         jerr = 8888 => negative response count y values
c         jerr = 9999 => no positive observations weights
c         jerr = 10000 => maxval(vp) <= 0.0
C      jerr < 0 => non fatal error - partial output:
c         Solutions for larger lamdas (1:(k-1)) returned.
c         jerr = -k => convergence for kth lamda value not reached
c            after maxit (see above) iterations.
c         jerr = -10000-k => number of non zero coefficients along path
c            exceeds nx (see above) at kth lamda value.
c    o(no) = training data values for last (lmu_th) solution linear
c            combination.
c
c
c Poisson utility routines:
c
c
c same as elnet above:
c
c    call solns(ni,nx,lmu,ca,ia,nin,b)
c    call uncomp(ni,ca,ia,nin,a)
c    call modval(a0,ca,ia,nin,n,x,f);
c    call cmodval(a0,ca,ia,nin,x,ix,jx,n,f);
c
c compute deviance for given uncompressed data and set of uncompressed
c solutions
c
c call deviance(no,ni,x,y,o,w,nsol,a0,a,flog,jerr)
c
c input:
c
c   no = number of observations
c   ni = number of predictor variables
c   x(no,ni) = predictor data matrix flat file
c   y(no) = observation response counts
c   o(no) = observation off-sets
c   w(no)= observation weights
c   nsol = number of solutions
c   a0(nsol) = intercept for each solution
c   a(ni,nsol) = solution coefficient vectors (uncompressed)
c
c output:
c
c   flog(nsol) = respective deviance values minus null deviance
c   jerr = error flag - see above
c
c
c compute deviance for given compressed data and set of uncompressed solutions
c
c call spdeviance(no,ni,x,ix,jx,y,o,w,nsol,a0,a,flog,jerr)
c
c input:
c
c   no = number of observations
c   ni = number of predictor variables
c   x, ix, jx = predictor data matrix in compressed sparse row format
c   y(no) = observation response counts
c   o(no) = observation off-sets
c   w(no)= observation weights
c   nsol = number of solutions
c   a0(nsol) = intercept for each solution
c   a(ni,nsol) = solution coefficient vectors (uncompressed)
c
c output
c
c   flog(nsol) = respective deviance values minus null deviance
c   jerr = error flag - see above
c
c
c compute deviance for given compressed data and compressed solutions
c
c call cspdeviance(no,x,ix,jx,y,o,w,nx,lmu,a0,ca,ia,nin,flog,jerr)
c
c input:
c
c   no = number of observations
c   x, ix, jx = predictor data matrix in compressed sparse row format
c   y(no) = observation response counts
c   o(no) = observation off-sets
c   w(no)= observation weights
c   nx = input to spfishnet
c   lmu,a0(lmu),ca(nx,lmu),ia(nx),nin(lmu) = output from spfishnet
c
c output
c
c   flog(lmu) = respective deviance values minus null deviance
c   jerr = error flag - see above
c
c
c
c          Elastic net with Cox proportional hazards model
c
c
c dense predictor matrix:
c
c call coxnet (parm,no,ni,x,y,d,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
c              maxit,isd,lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
c
c input:
c
c   x(no,ni) = predictor data matrix flat file (overwritten)
c   y(no) = observation times
c   d(no) = died/censored indicator
c       d(i)=0.0 => y(i) = censoring time
c       d(i)=1.0 => y(i) = death time
c   o(no) = observation off-sets
c   parm,no,ni,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,maxit
c                = same as fishnet above
c
c output:
c
c   lmu,ca,ia,nin,dev0,fdev,alm,nlp = same as fishnet above
c   jerr = error flag
c      jerr = 0  => no error - output returned
c      jerr > 0 => fatal error - no output returned
c         jerr < 7777 => memory allocation error
c         jerr = 7777 => all used predictors have zero variance
c         jerr = 8888 => all observations censored (d(i)=0.0)
c         jerr = 9999 => no positive observations weights
c         jerr = 10000 => maxval(vp) <= 0.0
c         jerr = 20000, 30000 => initialization numerical error
C      jerr < 0 => non fatal error - partial output:
c         Solutions for larger lamdas (1:(k-1)) returned.
c         jerr = -k => convergence for kth lamda value not reached
c            after maxit (see above) iterations.
c         jerr = -10000-k => number of non zero coefficients along path
c            exceeds nx (see above) at kth lamda value.
c         jerr = -30000-k => numerical error at kth lambda value
c    o(no) = training data values for last (lmu_th) solution linear
c            combination.
c
c
c
c coxnet utility routines:
c
c
c same as elnet above:
c
c    call solns(ni,nx,lmu,ca,ia,nin,b)
c    call uncomp(ni,ca,ia,nin,a)
c
c
c evaluate linear model from compressed coefficients and
c uncompressed predictor matrix:
c
c call cxmodval(ca,ia,nin,n,x,f);
c
c input:
c
c    ca(nx) = compressed coefficient values for a solution
c    ia(nx) = pointers to compressed coefficients
c    nin = number of compressed coefficients for solution
c    n = number of predictor vectors (observations)
c    x(n,ni) = full (uncompressed) predictor matrix
c
c output:
c
c    f(n) = model predictions
c
c
c compute log-likelihood for given data set and vectors of coefficients
c
c call loglike(no,ni,x,y,d,o,w,nvec,a,flog,jerr)
c
c input:
c
c   no = number of observations
c   ni = number of predictor variables
c   x(no,ni) = predictor data matrix flat file
c   y(no) = observation times
c   d(no) = died/censored indicator
c       d(i)=0.0 => y(i) = censoring time
c       d(i)=1.0 => y(i) = death time
c   o(no) = observation off-sets
c   w(no)= observation weights
c   nvec = number of coefficient vectors
c   a(ni,nvec) = coefficient vectors (uncompressed)
c
c output
c
c   flog(nvec) = respective log-likelihood values
c   jerr = error flag - see coxnet above
c
c
c
c
c                Changing internal parameter values
c
c
c call chg_fract_dev(fdev)
c   fdev = minimum fractional change in deviance for stopping path
c      default = 1.0e-5
c
c call chg_dev_max(devmax)
c   devmax = maximum fraction of explained deviance for stopping path
c      default = 0.999
c
c call chg_min_flmin(eps)
c   eps = minimum value of flmin (see above). default= 1.0e-6
c
c call chg_big(big)
c   big = large floating point number. default = 9.9e35
c
c call chg_min_lambdas(mnlam)
c   mnlam = minimum number of path points (lambda values) allowed
c      default = 5
c
c call chg_min_null_prob(pmin)
c   pmin = minimum null probability for any class. default = 1.0e-9
c
c call chg _max_exp(exmx)
c   exmx = maximum allowed exponent. default = 250.0
c
c call chg_bnorm(prec,mxit)
c   prec = convergence threshold for multi response bounds adjustment
c          solution. default = 1.0e-10.
c   mxit = maximum iterations for multiresponse bounds adjustment solution
c          default = 100.
c
c
c             Obtain current internal parameter values
c
c call get_int_parms(fdev,eps,big,mnlam,devmax,pmin,exmx)
c call get_bnorm(prec,mxit);
c
c
c             
      subroutine get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)          772
      data sml0,eps0,big0,mnlam0,rsqmax0,pmin0,exmx0  /1.0e-5,1.0e-6,9.9    774 
     *e35,5,0.999,1.0e-9,250.0/
      sml=sml0                                                              774
      eps=eps0                                                              774
      big=big0                                                              774
      mnlam=mnlam0                                                          774
      rsqmax=rsqmax0                                                        775
      pmin=pmin0                                                            775
      exmx=exmx0                                                            776
      return                                                                777
      entry chg_fract_dev(arg)                                              777
      sml0=arg                                                              777
      return                                                                778
      entry chg_dev_max(arg)                                                778
      rsqmax0=arg                                                           778
      return                                                                779
      entry chg_min_flmin(arg)                                              779
      eps0=arg                                                              779
      return                                                                780
      entry chg_big(arg)                                                    780
      big0=arg                                                              780
      return                                                                781
      entry chg_min_lambdas(irg)                                            781
      mnlam0=irg                                                            781
      return                                                                782
      entry chg_min_null_prob(arg)                                          782
      pmin0=arg                                                             782
      return                                                                783
      entry chg_max_exp(arg)                                                783
      exmx0=arg                                                             783
      return                                                                784
      end                                                                   785
      subroutine elnet  (ka,parm,no,ni,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,u    788 
     *lam,thr,isd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      real x(no,ni),y(no),w(no),vp(ni),ca(nx,nlam),cl(2,ni)                 789
      real ulam(nlam),a0(nlam),rsq(nlam),alm(nlam)                          790
      integer jd(*),ia(nx),nin(nlam)                                        791
      real, dimension (:), allocatable :: vq;                                   
      if(maxval(vp) .gt. 0.0)goto 10021                                     794
      jerr=10000                                                            794
      return                                                                794
10021 continue                                                              795
      allocate(vq(1:ni),stat=jerr)                                          795
      if(jerr.ne.0) return                                                  796
      vq=max(0.0,vp)                                                        796
      vq=vq*ni/sum(vq)                                                      797
      if(ka .ne. 1)goto 10041                                               798
      call elnetu  (parm,no,ni,x,y,w,jd,vq,cl,ne,nx,nlam,flmin,ulam,thr,    801 
     *isd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      goto 10051                                                            802
10041 continue                                                              803
      call elnetn (parm,no,ni,x,y,w,jd,vq,cl,ne,nx,nlam,flmin,ulam,thr,i    806 
     *sd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
10051 continue                                                              807
10031 continue                                                              807
      deallocate(vq)                                                        808
      return                                                                809
      end                                                                   810
      subroutine elnetu  (parm,no,ni,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,ula    813 
     *m,thr,isd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      real x(no,ni),y(no),w(no),vp(ni),ulam(nlam),cl(2,ni)                  814
      real ca(nx,nlam),a0(nlam),rsq(nlam),alm(nlam)                         815
      integer jd(*),ia(nx),nin(nlam)                                        816
      real, dimension (:), allocatable :: xm,xs,g,xv,vlam                       
      integer, dimension (:), allocatable :: ju                                 
      allocate(g(1:ni),stat=jerr)                                           821
      allocate(xm(1:ni),stat=ierr)                                          821
      jerr=jerr+ierr                                                        822
      allocate(xs(1:ni),stat=ierr)                                          822
      jerr=jerr+ierr                                                        823
      allocate(ju(1:ni),stat=ierr)                                          823
      jerr=jerr+ierr                                                        824
      allocate(xv(1:ni),stat=ierr)                                          824
      jerr=jerr+ierr                                                        825
      allocate(vlam(1:nlam),stat=ierr)                                      825
      jerr=jerr+ierr                                                        826
      if(jerr.ne.0) return                                                  827
      call chkvars(no,ni,x,ju)                                              828
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                  829
      if(maxval(ju) .gt. 0)goto 10071                                       829
      jerr=7777                                                             829
      return                                                                829
10071 continue                                                              830
      call standard(no,ni,x,y,w,isd,intr,ju,g,xm,xs,ym,ys,xv,jerr)          831
      if(jerr.ne.0) return                                                  832
      cl=cl/ys                                                              832
      if(isd .le. 0)goto 10091                                              832
10100 do 10101 j=1,ni                                                       832
      cl(:,j)=cl(:,j)*xs(j)                                                 832
10101 continue                                                              832
10102 continue                                                              832
10091 continue                                                              833
      if(flmin.ge.1.0) vlam=ulam/ys                                         834
      call elnet1(parm,ni,ju,vp,cl,g,no,ne,nx,x,nlam,flmin,vlam,thr,maxi    836 
     *t,xv,  lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                  837
10110 do 10111 k=1,lmu                                                      837
      alm(k)=ys*alm(k)                                                      837
      nk=nin(k)                                                             838
10120 do 10121 l=1,nk                                                       838
      ca(l,k)=ys*ca(l,k)/xs(ia(l))                                          838
10121 continue                                                              838
10122 continue                                                              838
      a0(k)=0.0                                                             839
      if(intr.ne.0) a0(k)=ym-dot_product(ca(1:nk,k),xm(ia(1:nk)))           840
10111 continue                                                              841
10112 continue                                                              841
      deallocate(xm,xs,g,ju,xv,vlam)                                        842
      return                                                                843
      end                                                                   844
      subroutine standard (no,ni,x,y,w,isd,intr,ju,g,xm,xs,ym,ys,xv,jerr    845 
     *)
      real x(no,ni),y(no),w(no),g(ni),xm(ni),xs(ni),xv(ni)                  845
      integer ju(ni)                                                        846
      real, dimension (:), allocatable :: v                                     
      allocate(v(1:no),stat=jerr)                                           849
      if(jerr.ne.0) return                                                  850
      w=w/sum(w)                                                            850
      v=sqrt(w)                                                             851
      if(intr .ne. 0)goto 10141                                             851
      ym=0.0                                                                851
      y=v*y                                                                 852
      ys=sqrt(dot_product(y,y)-dot_product(v,y)**2)                         852
      y=y/ys                                                                853
10150 do 10151 j=1,ni                                                       853
      if(ju(j).eq.0)goto 10151                                              853
      xm(j)=0.0                                                             853
      x(:,j)=v*x(:,j)                                                       854
      xv(j)=dot_product(x(:,j),x(:,j))                                      855
      if(isd .eq. 0)goto 10171                                              855
      xbq=dot_product(v,x(:,j))**2                                          855
      vc=xv(j)-xbq                                                          856
      xs(j)=sqrt(vc)                                                        856
      x(:,j)=x(:,j)/xs(j)                                                   856
      xv(j)=1.0+xbq/vc                                                      857
      goto 10181                                                            858
10171 continue                                                              858
      xs(j)=1.0                                                             858
10181 continue                                                              859
10161 continue                                                              859
10151 continue                                                              860
10152 continue                                                              860
      goto 10191                                                            861
10141 continue                                                              862
10200 do 10201 j=1,ni                                                       862
      if(ju(j).eq.0)goto 10201                                              863
      xm(j)=dot_product(w,x(:,j))                                           863
      x(:,j)=v*(x(:,j)-xm(j))                                               864
      xv(j)=dot_product(x(:,j),x(:,j))                                      864
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                        865
10201 continue                                                              866
10202 continue                                                              866
      if(isd .ne. 0)goto 10221                                              866
      xs=1.0                                                                866
      goto 10231                                                            867
10221 continue                                                              868
10240 do 10241 j=1,ni                                                       868
      if(ju(j).eq.0)goto 10241                                              868
      x(:,j)=x(:,j)/xs(j)                                                   868
10241 continue                                                              869
10242 continue                                                              869
      xv=1.0                                                                870
10231 continue                                                              871
10211 continue                                                              871
      ym=dot_product(w,y)                                                   871
      y=v*(y-ym)                                                            871
      ys=sqrt(dot_product(y,y))                                             871
      y=y/ys                                                                872
10191 continue                                                              873
10131 continue                                                              873
      g=0.0                                                                 873
10250 do 10251 j=1,ni                                                       873
      if(ju(j).ne.0) g(j)=dot_product(y,x(:,j))                             873
10251 continue                                                              874
10252 continue                                                              874
      deallocate(v)                                                         875
      return                                                                876
      end                                                                   877
      subroutine elnet1 (beta,ni,ju,vp,cl,g,no,ne,nx,x,nlam,flmin,ulam,t    879 
     *hr,maxit,xv,  lmu,ao,ia,kin,rsqo,almo,nlp,jerr)
      real vp(ni),g(ni),x(no,ni),ulam(nlam),ao(nx,nlam),rsqo(nlam),almo(    880 
     *nlam),xv(ni)
      real cl(2,ni)                                                         881
      integer ju(ni),ia(nx),kin(nlam)                                       882
      real, dimension (:), allocatable :: a,da                                  
      integer, dimension (:), allocatable :: mm                                 
      real, dimension (:,:), allocatable :: c                                   
      allocate(c(1:ni,1:nx),stat=jerr)                                          
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)                889
      allocate(a(1:ni),stat=ierr)                                           889
      jerr=jerr+ierr                                                        890
      allocate(mm(1:ni),stat=ierr)                                          890
      jerr=jerr+ierr                                                        891
      allocate(da(1:ni),stat=ierr)                                          891
      jerr=jerr+ierr                                                        892
      if(jerr.ne.0) return                                                  893
      bta=beta                                                              893
      omb=1.0-bta                                                           894
      if(flmin .ge. 1.0)goto 10271                                          894
      eqs=max(eps,flmin)                                                    894
      alf=eqs**(1.0/(nlam-1))                                               894
10271 continue                                                              895
      rsq=0.0                                                               895
      a=0.0                                                                 895
      mm=0                                                                  895
      nlp=0                                                                 895
      nin=nlp                                                               895
      iz=0                                                                  895
      mnl=min(mnlam,nlam)                                                   896
10280 do 10281 m=1,nlam                                                     897
      if(flmin .lt. 1.0)goto 10301                                          897
      alm=ulam(m)                                                           897
      goto 10291                                                            898
10301 if(m .le. 2)goto 10311                                                898
      alm=alm*alf                                                           898
      goto 10291                                                            899
10311 if(m .ne. 1)goto 10321                                                899
      alm=big                                                               899
      goto 10331                                                            900
10321 continue                                                              900
      alm=0.0                                                               901
10340 do 10341 j=1,ni                                                       901
      if(ju(j).eq.0)goto 10341                                              901
      if(vp(j).le.0.0)goto 10341                                            902
      alm=max(alm,abs(g(j))/vp(j))                                          903
10341 continue                                                              904
10342 continue                                                              904
      alm=alf*alm/max(bta,1.0e-3)                                           905
10331 continue                                                              906
10291 continue                                                              906
      dem=alm*omb                                                           906
      ab=alm*bta                                                            906
      rsq0=rsq                                                              906
      jz=1                                                                  907
10350 continue                                                              907
10351 continue                                                              907
      if(iz*jz.ne.0) go to 10360                                            907
      nlp=nlp+1                                                             907
      dlx=0.0                                                               908
10370 do 10371 k=1,ni                                                       908
      if(ju(k).eq.0)goto 10371                                              909
      ak=a(k)                                                               909
      u=g(k)+ak*xv(k)                                                       909
      v=abs(u)-vp(k)*ab                                                     909
      a(k)=0.0                                                              911
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d    912 
     *em)))
      if(a(k).eq.ak)goto 10371                                              913
      if(mm(k) .ne. 0)goto 10391                                            913
      nin=nin+1                                                             913
      if(nin.gt.nx)goto 10372                                               914
10400 do 10401 j=1,ni                                                       914
      if(ju(j).eq.0)goto 10401                                              915
      if(mm(j) .eq. 0)goto 10421                                            915
      c(j,nin)=c(k,mm(j))                                                   915
      goto 10401                                                            915
10421 continue                                                              916
      if(j .ne. k)goto 10441                                                916
      c(j,nin)=xv(j)                                                        916
      goto 10401                                                            916
10441 continue                                                              917
      c(j,nin)=dot_product(x(:,j),x(:,k))                                   918
10401 continue                                                              919
10402 continue                                                              919
      mm(k)=nin                                                             919
      ia(nin)=k                                                             920
10391 continue                                                              921
      del=a(k)-ak                                                           921
      rsq=rsq+del*(2.0*g(k)-del*xv(k))                                      922
      dlx=max(xv(k)*del**2,dlx)                                             923
10450 do 10451 j=1,ni                                                       923
      if(ju(j).ne.0) g(j)=g(j)-c(j,mm(k))*del                               923
10451 continue                                                              924
10452 continue                                                              924
10371 continue                                                              925
10372 continue                                                              925
      if(dlx.lt.thr)goto 10352                                              925
      if(nin.gt.nx)goto 10352                                               926
      if(nlp .le. maxit)goto 10471                                          926
      jerr=-m                                                               926
      return                                                                926
10471 continue                                                              927
10360 continue                                                              927
      iz=1                                                                  927
      da(1:nin)=a(ia(1:nin))                                                928
10480 continue                                                              928
10481 continue                                                              928
      nlp=nlp+1                                                             928
      dlx=0.0                                                               929
10490 do 10491 l=1,nin                                                      929
      k=ia(l)                                                               929
      ak=a(k)                                                               929
      u=g(k)+ak*xv(k)                                                       929
      v=abs(u)-vp(k)*ab                                                     930
      a(k)=0.0                                                              932
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d    933 
     *em)))
      if(a(k).eq.ak)goto 10491                                              934
      del=a(k)-ak                                                           934
      rsq=rsq+del*(2.0*g(k)-del*xv(k))                                      935
      dlx=max(xv(k)*del**2,dlx)                                             936
10500 do 10501 j=1,nin                                                      936
      g(ia(j))=g(ia(j))-c(ia(j),mm(k))*del                                  936
10501 continue                                                              937
10502 continue                                                              937
10491 continue                                                              938
10492 continue                                                              938
      if(dlx.lt.thr)goto 10482                                              938
      if(nlp .le. maxit)goto 10521                                          938
      jerr=-m                                                               938
      return                                                                938
10521 continue                                                              939
      goto 10481                                                            940
10482 continue                                                              940
      da(1:nin)=a(ia(1:nin))-da(1:nin)                                      941
10530 do 10531 j=1,ni                                                       941
      if(mm(j).ne.0)goto 10531                                              942
      if(ju(j).ne.0) g(j)=g(j)-dot_product(da(1:nin),c(j,1:nin))            943
10531 continue                                                              944
10532 continue                                                              944
      jz=0                                                                  945
      goto 10351                                                            946
10352 continue                                                              946
      if(nin .le. nx)goto 10551                                             946
      jerr=-10000-m                                                         946
      goto 10282                                                            946
10551 continue                                                              947
      if(nin.gt.0) ao(1:nin,m)=a(ia(1:nin))                                 947
      kin(m)=nin                                                            948
      rsqo(m)=rsq                                                           948
      almo(m)=alm                                                           948
      lmu=m                                                                 949
      if(m.lt.mnl)goto 10281                                                949
      if(flmin.ge.1.0)goto 10281                                            950
      me=0                                                                  950
10560 do 10561 j=1,nin                                                      950
      if(ao(j,m).ne.0.0) me=me+1                                            950
10561 continue                                                              950
10562 continue                                                              950
      if(me.gt.ne)goto 10282                                                951
      if(rsq-rsq0.lt.sml*rsq)goto 10282                                     951
      if(rsq.gt.rsqmax)goto 10282                                           952
10281 continue                                                              953
10282 continue                                                              953
      deallocate(a,mm,c,da)                                                 954
      return                                                                955
      end                                                                   956
      subroutine elnetn (parm,no,ni,x,y,w,jd,vp,cl,ne,nx,nlam,flmin,ulam    958 
     *,thr,isd,  intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      real vp(ni),x(no,ni),y(no),w(no),ulam(nlam),cl(2,ni)                  959
      real ca(nx,nlam),a0(nlam),rsq(nlam),alm(nlam)                         960
      integer jd(*),ia(nx),nin(nlam)                                        961
      real, dimension (:), allocatable :: xm,xs,xv,vlam                         
      integer, dimension (:), allocatable :: ju                                 
      allocate(xm(1:ni),stat=jerr)                                          966
      allocate(xs(1:ni),stat=ierr)                                          966
      jerr=jerr+ierr                                                        967
      allocate(ju(1:ni),stat=ierr)                                          967
      jerr=jerr+ierr                                                        968
      allocate(xv(1:ni),stat=ierr)                                          968
      jerr=jerr+ierr                                                        969
      allocate(vlam(1:nlam),stat=ierr)                                      969
      jerr=jerr+ierr                                                        970
      if(jerr.ne.0) return                                                  971
      call chkvars(no,ni,x,ju)                                              972
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                  973
      if(maxval(ju) .gt. 0)goto 10581                                       973
      jerr=7777                                                             973
      return                                                                973
10581 continue                                                              974
      call standard1(no,ni,x,y,w,isd,intr,ju,xm,xs,ym,ys,xv,jerr)           975
      if(jerr.ne.0) return                                                  976
      cl=cl/ys                                                              976
      if(isd .le. 0)goto 10601                                              976
10610 do 10611 j=1,ni                                                       976
      cl(:,j)=cl(:,j)*xs(j)                                                 976
10611 continue                                                              976
10612 continue                                                              976
10601 continue                                                              977
      if(flmin.ge.1.0) vlam=ulam/ys                                         978
      call elnet2(parm,ni,ju,vp,cl,y,no,ne,nx,x,nlam,flmin,vlam,thr,maxi    980 
     *t,xv,  lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                  981
10620 do 10621 k=1,lmu                                                      981
      alm(k)=ys*alm(k)                                                      981
      nk=nin(k)                                                             982
10630 do 10631 l=1,nk                                                       982
      ca(l,k)=ys*ca(l,k)/xs(ia(l))                                          982
10631 continue                                                              982
10632 continue                                                              982
      a0(k)=0.0                                                             983
      if(intr.ne.0) a0(k)=ym-dot_product(ca(1:nk,k),xm(ia(1:nk)))           984
10621 continue                                                              985
10622 continue                                                              985
      deallocate(xm,xs,ju,xv,vlam)                                          986
      return                                                                987
      end                                                                   988
      subroutine standard1 (no,ni,x,y,w,isd,intr,ju,xm,xs,ym,ys,xv,jerr)    989
      real x(no,ni),y(no),w(no),xm(ni),xs(ni),xv(ni)                        989
      integer ju(ni)                                                        990
      real, dimension (:), allocatable :: v                                     
      allocate(v(1:no),stat=jerr)                                           993
      if(jerr.ne.0) return                                                  994
      w=w/sum(w)                                                            994
      v=sqrt(w)                                                             995
      if(intr .ne. 0)goto 10651                                             995
      ym=0.0                                                                995
      y=v*y                                                                 996
      ys=sqrt(dot_product(y,y)-dot_product(v,y)**2)                         996
      y=y/ys                                                                997
10660 do 10661 j=1,ni                                                       997
      if(ju(j).eq.0)goto 10661                                              997
      xm(j)=0.0                                                             997
      x(:,j)=v*x(:,j)                                                       998
      xv(j)=dot_product(x(:,j),x(:,j))                                      999
      if(isd .eq. 0)goto 10681                                              999
      xbq=dot_product(v,x(:,j))**2                                          999
      vc=xv(j)-xbq                                                         1000
      xs(j)=sqrt(vc)                                                       1000
      x(:,j)=x(:,j)/xs(j)                                                  1000
      xv(j)=1.0+xbq/vc                                                     1001
      goto 10691                                                           1002
10681 continue                                                             1002
      xs(j)=1.0                                                            1002
10691 continue                                                             1003
10671 continue                                                             1003
10661 continue                                                             1004
10662 continue                                                             1004
      go to 10700                                                          1005
10651 continue                                                             1006
10710 do 10711 j=1,ni                                                      1006
      if(ju(j).eq.0)goto 10711                                             1007
      xm(j)=dot_product(w,x(:,j))                                          1007
      x(:,j)=v*(x(:,j)-xm(j))                                              1008
      xv(j)=dot_product(x(:,j),x(:,j))                                     1008
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                       1009
10711 continue                                                             1010
10712 continue                                                             1010
      if(isd .ne. 0)goto 10731                                             1010
      xs=1.0                                                               1010
      goto 10741                                                           1011
10731 continue                                                             1011
10750 do 10751 j=1,ni                                                      1011
      if(ju(j).eq.0)goto 10751                                             1011
      x(:,j)=x(:,j)/xs(j)                                                  1011
10751 continue                                                             1012
10752 continue                                                             1012
      xv=1.0                                                               1013
10741 continue                                                             1014
10721 continue                                                             1014
      ym=dot_product(w,y)                                                  1014
      y=v*(y-ym)                                                           1014
      ys=sqrt(dot_product(y,y))                                            1014
      y=y/ys                                                               1015
10700 continue                                                             1015
      deallocate(v)                                                        1016
      return                                                               1017
      end                                                                  1018
      subroutine elnet2(beta,ni,ju,vp,cl,y,no,ne,nx,x,nlam,flmin,ulam,th   1020 
     *r,maxit,xv,  lmu,ao,ia,kin,rsqo,almo,nlp,jerr)
      real vp(ni),y(no),x(no,ni),ulam(nlam),ao(nx,nlam),rsqo(nlam),almo(   1021 
     *nlam),xv(ni)
      real cl(2,ni)                                                        1022
      integer ju(ni),ia(nx),kin(nlam)                                      1023
      real, dimension (:), allocatable :: a,g                                   
      integer, dimension (:), allocatable :: mm,ix                              
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)               1028
      allocate(a(1:ni),stat=jerr)                                          1029
      allocate(mm(1:ni),stat=ierr)                                         1029
      jerr=jerr+ierr                                                       1030
      allocate(g(1:ni),stat=ierr)                                          1030
      jerr=jerr+ierr                                                       1031
      allocate(ix(1:ni),stat=ierr)                                         1031
      jerr=jerr+ierr                                                       1032
      if(jerr.ne.0) return                                                 1033
      bta=beta                                                             1033
      omb=1.0-bta                                                          1033
      ix=0                                                                 1034
      if(flmin .ge. 1.0)goto 10771                                         1034
      eqs=max(eps,flmin)                                                   1034
      alf=eqs**(1.0/(nlam-1))                                              1034
10771 continue                                                             1035
      rsq=0.0                                                              1035
      a=0.0                                                                1035
      mm=0                                                                 1035
      nlp=0                                                                1035
      nin=nlp                                                              1035
      iz=0                                                                 1035
      mnl=min(mnlam,nlam)                                                  1035
      alm=0.0                                                              1036
10780 do 10781 j=1,ni                                                      1036
      if(ju(j).eq.0)goto 10781                                             1036
      g(j)=abs(dot_product(y,x(:,j)))                                      1036
10781 continue                                                             1037
10782 continue                                                             1037
10790 do 10791 m=1,nlam                                                    1037
      alm0=alm                                                             1038
      if(flmin .lt. 1.0)goto 10811                                         1038
      alm=ulam(m)                                                          1038
      goto 10801                                                           1039
10811 if(m .le. 2)goto 10821                                               1039
      alm=alm*alf                                                          1039
      goto 10801                                                           1040
10821 if(m .ne. 1)goto 10831                                               1040
      alm=big                                                              1040
      goto 10841                                                           1041
10831 continue                                                             1041
      alm0=0.0                                                             1042
10850 do 10851 j=1,ni                                                      1042
      if(ju(j).eq.0)goto 10851                                             1042
      if(vp(j).gt.0.0) alm0=max(alm0,g(j)/vp(j))                           1042
10851 continue                                                             1043
10852 continue                                                             1043
      alm0=alm0/max(bta,1.0e-3)                                            1043
      alm=alf*alm0                                                         1044
10841 continue                                                             1045
10801 continue                                                             1045
      dem=alm*omb                                                          1045
      ab=alm*bta                                                           1045
      rsq0=rsq                                                             1045
      jz=1                                                                 1046
      tlam=bta*(2.0*alm-alm0)                                              1047
10860 do 10861 k=1,ni                                                      1047
      if(ix(k).eq.1)goto 10861                                             1047
      if(ju(k).eq.0)goto 10861                                             1048
      if(g(k).gt.tlam*vp(k)) ix(k)=1                                       1049
10861 continue                                                             1050
10862 continue                                                             1050
10870 continue                                                             1050
10871 continue                                                             1050
      if(iz*jz.ne.0) go to 10360                                           1051
10880 continue                                                             1051
      nlp=nlp+1                                                            1051
      dlx=0.0                                                              1052
10890 do 10891 k=1,ni                                                      1052
      if(ix(k).eq.0)goto 10891                                             1052
      gk=dot_product(y,x(:,k))                                             1053
      ak=a(k)                                                              1053
      u=gk+ak*xv(k)                                                        1053
      v=abs(u)-vp(k)*ab                                                    1053
      a(k)=0.0                                                             1055
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1056 
     *em)))
      if(a(k).eq.ak)goto 10891                                             1057
      if(mm(k) .ne. 0)goto 10911                                           1057
      nin=nin+1                                                            1057
      if(nin.gt.nx)goto 10892                                              1058
      mm(k)=nin                                                            1058
      ia(nin)=k                                                            1059
10911 continue                                                             1060
      del=a(k)-ak                                                          1060
      rsq=rsq+del*(2.0*gk-del*xv(k))                                       1061
      y=y-del*x(:,k)                                                       1061
      dlx=max(xv(k)*del**2,dlx)                                            1062
10891 continue                                                             1063
10892 continue                                                             1063
      if(nin.gt.nx)goto 10872                                              1064
      if(dlx .ge. thr)goto 10931                                           1064
      ixx=0                                                                1065
10940 do 10941 k=1,ni                                                      1065
      if(ix(k).eq.1)goto 10941                                             1065
      if(ju(k).eq.0)goto 10941                                             1066
      g(k)=abs(dot_product(y,x(:,k)))                                      1067
      if(g(k) .le. ab*vp(k))goto 10961                                     1067
      ix(k)=1                                                              1067
      ixx=1                                                                1067
10961 continue                                                             1068
10941 continue                                                             1069
10942 continue                                                             1069
      if(ixx.eq.1) go to 10880                                             1070
      goto 10872                                                           1071
10931 continue                                                             1072
      if(nlp .le. maxit)goto 10981                                         1072
      jerr=-m                                                              1072
      return                                                               1072
10981 continue                                                             1073
10360 continue                                                             1073
      iz=1                                                                 1074
10990 continue                                                             1074
10991 continue                                                             1074
      nlp=nlp+1                                                            1074
      dlx=0.0                                                              1075
11000 do 11001 l=1,nin                                                     1075
      k=ia(l)                                                              1075
      gk=dot_product(y,x(:,k))                                             1076
      ak=a(k)                                                              1076
      u=gk+ak*xv(k)                                                        1076
      v=abs(u)-vp(k)*ab                                                    1076
      a(k)=0.0                                                             1078
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1079 
     *em)))
      if(a(k).eq.ak)goto 11001                                             1080
      del=a(k)-ak                                                          1080
      rsq=rsq+del*(2.0*gk-del*xv(k))                                       1081
      y=y-del*x(:,k)                                                       1081
      dlx=max(xv(k)*del**2,dlx)                                            1082
11001 continue                                                             1083
11002 continue                                                             1083
      if(dlx.lt.thr)goto 10992                                             1083
      if(nlp .le. maxit)goto 11021                                         1083
      jerr=-m                                                              1083
      return                                                               1083
11021 continue                                                             1084
      goto 10991                                                           1085
10992 continue                                                             1085
      jz=0                                                                 1086
      goto 10871                                                           1087
10872 continue                                                             1087
      if(nin .le. nx)goto 11041                                            1087
      jerr=-10000-m                                                        1087
      goto 10792                                                           1087
11041 continue                                                             1088
      if(nin.gt.0) ao(1:nin,m)=a(ia(1:nin))                                1088
      kin(m)=nin                                                           1089
      rsqo(m)=rsq                                                          1089
      almo(m)=alm                                                          1089
      lmu=m                                                                1090
      if(m.lt.mnl)goto 10791                                               1090
      if(flmin.ge.1.0)goto 10791                                           1091
      me=0                                                                 1091
11050 do 11051 j=1,nin                                                     1091
      if(ao(j,m).ne.0.0) me=me+1                                           1091
11051 continue                                                             1091
11052 continue                                                             1091
      if(me.gt.ne)goto 10792                                               1092
      if(rsq-rsq0.lt.sml*rsq)goto 10792                                    1092
      if(rsq.gt.rsqmax)goto 10792                                          1093
10791 continue                                                             1094
10792 continue                                                             1094
      deallocate(a,mm,g,ix)                                                1095
      return                                                               1096
      end                                                                  1097
      subroutine chkvars(no,ni,x,ju)                                       1098
      real x(no,ni)                                                        1098
      integer ju(ni)                                                       1099
11060 do 11061 j=1,ni                                                      1099
      ju(j)=0                                                              1099
      t=x(1,j)                                                             1100
11070 do 11071 i=2,no                                                      1100
      if(x(i,j).eq.t)goto 11071                                            1100
      ju(j)=1                                                              1100
      goto 11072                                                           1100
11071 continue                                                             1101
11072 continue                                                             1101
11061 continue                                                             1102
11062 continue                                                             1102
      return                                                               1103
      end                                                                  1104
      subroutine uncomp(ni,ca,ia,nin,a)                                    1105
      real ca(*),a(ni)                                                     1105
      integer ia(*)                                                        1106
      a=0.0                                                                1106
      if(nin.gt.0) a(ia(1:nin))=ca(1:nin)                                  1107
      return                                                               1108
      end                                                                  1109
      subroutine modval(a0,ca,ia,nin,n,x,f)                                1110
      real ca(nin),x(n,*),f(n)                                             1110
      integer ia(nin)                                                      1111
      f=a0                                                                 1111
      if(nin.le.0) return                                                  1112
11080 do 11081 i=1,n                                                       1112
      f(i)=f(i)+dot_product(ca(1:nin),x(i,ia(1:nin)))                      1112
11081 continue                                                             1113
11082 continue                                                             1113
      return                                                               1114
      end                                                                  1115
      subroutine spelnet  (ka,parm,no,ni,x,ix,jx,y,w,jd,vp,cl,ne,nx,nlam   1118 
     *,flmin,ulam,thr,isd,intr,  maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr
     *)
      real x(*),y(no),w(no),vp(ni),ulam(nlam),cl(2,ni)                     1119
      real ca(nx,nlam),a0(nlam),rsq(nlam),alm(nlam)                        1120
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           1121
      real, dimension (:), allocatable :: vq;                                   
      if(maxval(vp) .gt. 0.0)goto 11101                                    1124
      jerr=10000                                                           1124
      return                                                               1124
11101 continue                                                             1125
      allocate(vq(1:ni),stat=jerr)                                         1125
      if(jerr.ne.0) return                                                 1126
      vq=max(0.0,vp)                                                       1126
      vq=vq*ni/sum(vq)                                                     1127
      if(ka .ne. 1)goto 11121                                              1128
      call spelnetu  (parm,no,ni,x,ix,jx,y,w,jd,vq,cl,ne,nx,nlam,flmin,u   1131 
     *lam,thr,isd,  intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      goto 11131                                                           1132
11121 continue                                                             1133
      call spelnetn (parm,no,ni,x,ix,jx,y,w,jd,vq,cl,ne,nx,nlam,flmin,ul   1136 
     *am,thr,isd,intr,  maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
11131 continue                                                             1137
11111 continue                                                             1137
      deallocate(vq)                                                       1138
      return                                                               1139
      end                                                                  1140
      subroutine spelnetu  (parm,no,ni,x,ix,jx,y,w,jd,vp,cl,ne,nx,nlam,f   1143 
     *lmin,ulam,thr,isd,intr,  maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      real x(*),y(no),w(no),vp(ni),ulam(nlam),cl(2,ni)                     1144
      real ca(nx,nlam),a0(nlam),rsq(nlam),alm(nlam)                        1145
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           1146
      real, dimension (:), allocatable :: xm,xs,g,xv,vlam                       
      integer, dimension (:), allocatable :: ju                                 
      allocate(g(1:ni),stat=jerr)                                          1151
      allocate(xm(1:ni),stat=ierr)                                         1151
      jerr=jerr+ierr                                                       1152
      allocate(xs(1:ni),stat=ierr)                                         1152
      jerr=jerr+ierr                                                       1153
      allocate(ju(1:ni),stat=ierr)                                         1153
      jerr=jerr+ierr                                                       1154
      allocate(xv(1:ni),stat=ierr)                                         1154
      jerr=jerr+ierr                                                       1155
      allocate(vlam(1:nlam),stat=ierr)                                     1155
      jerr=jerr+ierr                                                       1156
      if(jerr.ne.0) return                                                 1157
      call spchkvars(no,ni,x,ix,ju)                                        1158
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 1159
      if(maxval(ju) .gt. 0)goto 11151                                      1159
      jerr=7777                                                            1159
      return                                                               1159
11151 continue                                                             1160
      call spstandard(no,ni,x,ix,jx,y,w,ju,isd,intr,g,xm,xs,ym,ys,xv,jer   1161 
     *r)
      if(jerr.ne.0) return                                                 1162
      cl=cl/ys                                                             1162
      if(isd .le. 0)goto 11171                                             1162
11180 do 11181 j=1,ni                                                      1162
      cl(:,j)=cl(:,j)*xs(j)                                                1162
11181 continue                                                             1162
11182 continue                                                             1162
11171 continue                                                             1163
      if(flmin.ge.1.0) vlam=ulam/ys                                        1164
      call spelnet1(parm,ni,g,no,w,ne,nx,x,ix,jx,ju,vp,cl,nlam,flmin,vla   1166 
     *m,thr,maxit,  xm,xs,xv,lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                 1167
11190 do 11191 k=1,lmu                                                     1167
      alm(k)=ys*alm(k)                                                     1167
      nk=nin(k)                                                            1168
11200 do 11201 l=1,nk                                                      1168
      ca(l,k)=ys*ca(l,k)/xs(ia(l))                                         1168
11201 continue                                                             1168
11202 continue                                                             1168
      a0(k)=0.0                                                            1169
      if(intr.ne.0) a0(k)=ym-dot_product(ca(1:nk,k),xm(ia(1:nk)))          1170
11191 continue                                                             1171
11192 continue                                                             1171
      deallocate(xm,xs,g,ju,xv,vlam)                                       1172
      return                                                               1173
      end                                                                  1174
      subroutine spstandard (no,ni,x,ix,jx,y,w,ju,isd,intr,g,xm,xs,ym,ys   1175 
     *,xv,jerr)
      real x(*),y(no),w(no),g(ni),xm(ni),xs(ni),xv(ni)                     1175
      integer ix(*),jx(*),ju(ni)                                           1176
      w=w/sum(w)                                                           1177
      if(intr .ne. 0)goto 11221                                            1177
      ym=0.0                                                               1178
      ys=sqrt(dot_product(w,y**2)-dot_product(w,y)**2)                     1178
      y=y/ys                                                               1179
11230 do 11231 j=1,ni                                                      1179
      if(ju(j).eq.0)goto 11231                                             1179
      xm(j)=0.0                                                            1179
      jb=ix(j)                                                             1179
      je=ix(j+1)-1                                                         1180
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)                          1181
      if(isd .eq. 0)goto 11251                                             1181
      xbq=dot_product(w(jx(jb:je)),x(jb:je))**2                            1181
      vc=xv(j)-xbq                                                         1182
      xs(j)=sqrt(vc)                                                       1182
      xv(j)=1.0+xbq/vc                                                     1183
      goto 11261                                                           1184
11251 continue                                                             1184
      xs(j)=1.0                                                            1184
11261 continue                                                             1185
11241 continue                                                             1185
11231 continue                                                             1186
11232 continue                                                             1186
      goto 11271                                                           1187
11221 continue                                                             1188
11280 do 11281 j=1,ni                                                      1188
      if(ju(j).eq.0)goto 11281                                             1189
      jb=ix(j)                                                             1189
      je=ix(j+1)-1                                                         1189
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             1190
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j)**2                 1191
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                       1192
11281 continue                                                             1193
11282 continue                                                             1193
      if(isd .ne. 0)goto 11301                                             1193
      xs=1.0                                                               1193
      goto 11311                                                           1193
11301 continue                                                             1193
      xv=1.0                                                               1193
11311 continue                                                             1194
11291 continue                                                             1194
      ym=dot_product(w,y)                                                  1194
      y=y-ym                                                               1194
      ys=sqrt(dot_product(w,y**2))                                         1194
      y=y/ys                                                               1195
11271 continue                                                             1196
11211 continue                                                             1196
      g=0.0                                                                1197
11320 do 11321 j=1,ni                                                      1197
      if(ju(j).eq.0)goto 11321                                             1197
      jb=ix(j)                                                             1197
      je=ix(j+1)-1                                                         1198
      g(j)=dot_product(w(jx(jb:je))*y(jx(jb:je)),x(jb:je))/xs(j)           1199
11321 continue                                                             1200
11322 continue                                                             1200
      return                                                               1201
      end                                                                  1202
      subroutine spelnet1(beta,ni,g,no,w,ne,nx,x,ix,jx,ju,vp,cl,nlam,flm   1204 
     *in,ulam,  thr,maxit,xm,xs,xv,lmu,ao,ia,kin,rsqo,almo,nlp,jerr)
      real g(ni),vp(ni),x(*),ulam(nlam),w(no)                              1205
      real ao(nx,nlam),rsqo(nlam),almo(nlam),xm(ni),xs(ni),xv(ni),cl(2,n   1206 
     *i)
      integer ix(*),jx(*),ju(ni),ia(nx),kin(nlam)                          1207
      real, dimension (:), allocatable :: a,da                                  
      integer, dimension (:), allocatable :: mm                                 
      real, dimension (:,:), allocatable :: c                                   
      allocate(c(1:ni,1:nx),stat=jerr)                                          
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)               1214
      allocate(a(1:ni),stat=ierr)                                          1214
      jerr=jerr+ierr                                                       1215
      allocate(mm(1:ni),stat=ierr)                                         1215
      jerr=jerr+ierr                                                       1216
      allocate(da(1:ni),stat=ierr)                                         1216
      jerr=jerr+ierr                                                       1217
      if(jerr.ne.0) return                                                 1218
      bta=beta                                                             1218
      omb=1.0-bta                                                          1219
      if(flmin .ge. 1.0)goto 11341                                         1219
      eqs=max(eps,flmin)                                                   1219
      alf=eqs**(1.0/(nlam-1))                                              1219
11341 continue                                                             1220
      rsq=0.0                                                              1220
      a=0.0                                                                1220
      mm=0                                                                 1220
      nlp=0                                                                1220
      nin=nlp                                                              1220
      iz=0                                                                 1220
      mnl=min(mnlam,nlam)                                                  1221
11350 do 11351 m=1,nlam                                                    1222
      if(flmin .lt. 1.0)goto 11371                                         1222
      alm=ulam(m)                                                          1222
      goto 11361                                                           1223
11371 if(m .le. 2)goto 11381                                               1223
      alm=alm*alf                                                          1223
      goto 11361                                                           1224
11381 if(m .ne. 1)goto 11391                                               1224
      alm=big                                                              1224
      goto 11401                                                           1225
11391 continue                                                             1225
      alm=0.0                                                              1226
11410 do 11411 j=1,ni                                                      1226
      if(ju(j).eq.0)goto 11411                                             1226
      if(vp(j).le.0.0)goto 11411                                           1227
      alm=max(alm,abs(g(j))/vp(j))                                         1228
11411 continue                                                             1229
11412 continue                                                             1229
      alm=alf*alm/max(bta,1.0e-3)                                          1230
11401 continue                                                             1231
11361 continue                                                             1231
      dem=alm*omb                                                          1231
      ab=alm*bta                                                           1231
      rsq0=rsq                                                             1231
      jz=1                                                                 1232
11420 continue                                                             1232
11421 continue                                                             1232
      if(iz*jz.ne.0) go to 10360                                           1232
      nlp=nlp+1                                                            1232
      dlx=0.0                                                              1233
11430 do 11431 k=1,ni                                                      1233
      if(ju(k).eq.0)goto 11431                                             1234
      ak=a(k)                                                              1234
      u=g(k)+ak*xv(k)                                                      1234
      v=abs(u)-vp(k)*ab                                                    1234
      a(k)=0.0                                                             1236
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1237 
     *em)))
      if(a(k).eq.ak)goto 11431                                             1238
      if(mm(k) .ne. 0)goto 11451                                           1238
      nin=nin+1                                                            1238
      if(nin.gt.nx)goto 11432                                              1239
11460 do 11461 j=1,ni                                                      1239
      if(ju(j).eq.0)goto 11461                                             1240
      if(mm(j) .eq. 0)goto 11481                                           1240
      c(j,nin)=c(k,mm(j))                                                  1240
      goto 11461                                                           1240
11481 continue                                                             1241
      if(j .ne. k)goto 11501                                               1241
      c(j,nin)=xv(j)                                                       1241
      goto 11461                                                           1241
11501 continue                                                             1242
      c(j,nin)=  (row_prod(j,k,ix,jx,x,w)-xm(j)*xm(k))/(xs(j)*xs(k))       1244
11461 continue                                                             1245
11462 continue                                                             1245
      mm(k)=nin                                                            1245
      ia(nin)=k                                                            1246
11451 continue                                                             1247
      del=a(k)-ak                                                          1247
      rsq=rsq+del*(2.0*g(k)-del*xv(k))                                     1248
      dlx=max(xv(k)*del**2,dlx)                                            1249
11510 do 11511 j=1,ni                                                      1249
      if(ju(j).ne.0) g(j)=g(j)-c(j,mm(k))*del                              1249
11511 continue                                                             1250
11512 continue                                                             1250
11431 continue                                                             1251
11432 continue                                                             1251
      if(dlx.lt.thr)goto 11422                                             1251
      if(nin.gt.nx)goto 11422                                              1252
      if(nlp .le. maxit)goto 11531                                         1252
      jerr=-m                                                              1252
      return                                                               1252
11531 continue                                                             1253
10360 continue                                                             1253
      iz=1                                                                 1253
      da(1:nin)=a(ia(1:nin))                                               1254
11540 continue                                                             1254
11541 continue                                                             1254
      nlp=nlp+1                                                            1254
      dlx=0.0                                                              1255
11550 do 11551 l=1,nin                                                     1255
      k=ia(l)                                                              1256
      ak=a(k)                                                              1256
      u=g(k)+ak*xv(k)                                                      1256
      v=abs(u)-vp(k)*ab                                                    1256
      a(k)=0.0                                                             1258
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1259 
     *em)))
      if(a(k).eq.ak)goto 11551                                             1260
      del=a(k)-ak                                                          1260
      rsq=rsq+del*(2.0*g(k)-del*xv(k))                                     1261
      dlx=max(xv(k)*del**2,dlx)                                            1262
11560 do 11561 j=1,nin                                                     1262
      g(ia(j))=g(ia(j))-c(ia(j),mm(k))*del                                 1262
11561 continue                                                             1263
11562 continue                                                             1263
11551 continue                                                             1264
11552 continue                                                             1264
      if(dlx.lt.thr)goto 11542                                             1264
      if(nlp .le. maxit)goto 11581                                         1264
      jerr=-m                                                              1264
      return                                                               1264
11581 continue                                                             1265
      goto 11541                                                           1266
11542 continue                                                             1266
      da(1:nin)=a(ia(1:nin))-da(1:nin)                                     1267
11590 do 11591 j=1,ni                                                      1267
      if(mm(j).ne.0)goto 11591                                             1268
      if(ju(j).ne.0) g(j)=g(j)-dot_product(da(1:nin),c(j,1:nin))           1269
11591 continue                                                             1270
11592 continue                                                             1270
      jz=0                                                                 1271
      goto 11421                                                           1272
11422 continue                                                             1272
      if(nin .le. nx)goto 11611                                            1272
      jerr=-10000-m                                                        1272
      goto 11352                                                           1272
11611 continue                                                             1273
      if(nin.gt.0) ao(1:nin,m)=a(ia(1:nin))                                1273
      kin(m)=nin                                                           1274
      rsqo(m)=rsq                                                          1274
      almo(m)=alm                                                          1274
      lmu=m                                                                1275
      if(m.lt.mnl)goto 11351                                               1275
      if(flmin.ge.1.0)goto 11351                                           1276
      me=0                                                                 1276
11620 do 11621 j=1,nin                                                     1276
      if(ao(j,m).ne.0.0) me=me+1                                           1276
11621 continue                                                             1276
11622 continue                                                             1276
      if(me.gt.ne)goto 11352                                               1277
      if(rsq-rsq0.lt.sml*rsq)goto 11352                                    1277
      if(rsq.gt.rsqmax)goto 11352                                          1278
11351 continue                                                             1279
11352 continue                                                             1279
      deallocate(a,mm,c,da)                                                1280
      return                                                               1281
      end                                                                  1282
      subroutine spelnetn(parm,no,ni,x,ix,jx,y,w,jd,vp,cl,ne,nx,nlam,flm   1284 
     *in,ulam,  thr,isd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      real x(*),vp(ni),y(no),w(no),ulam(nlam),cl(2,ni)                     1285
      real ca(nx,nlam),a0(nlam),rsq(nlam),alm(nlam)                        1286
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           1287
      real, dimension (:), allocatable :: xm,xs,xv,vlam                         
      integer, dimension (:), allocatable :: ju                                 
      allocate(xm(1:ni),stat=jerr)                                         1292
      allocate(xs(1:ni),stat=ierr)                                         1292
      jerr=jerr+ierr                                                       1293
      allocate(ju(1:ni),stat=ierr)                                         1293
      jerr=jerr+ierr                                                       1294
      allocate(xv(1:ni),stat=ierr)                                         1294
      jerr=jerr+ierr                                                       1295
      allocate(vlam(1:nlam),stat=ierr)                                     1295
      jerr=jerr+ierr                                                       1296
      if(jerr.ne.0) return                                                 1297
      call spchkvars(no,ni,x,ix,ju)                                        1298
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 1299
      if(maxval(ju) .gt. 0)goto 11641                                      1299
      jerr=7777                                                            1299
      return                                                               1299
11641 continue                                                             1300
      call spstandard1(no,ni,x,ix,jx,y,w,ju,isd,intr,xm,xs,ym,ys,xv,jerr   1301 
     *)
      if(jerr.ne.0) return                                                 1302
      cl=cl/ys                                                             1302
      if(isd .le. 0)goto 11661                                             1302
11670 do 11671 j=1,ni                                                      1302
      cl(:,j)=cl(:,j)*xs(j)                                                1302
11671 continue                                                             1302
11672 continue                                                             1302
11661 continue                                                             1303
      if(flmin.ge.1.0) vlam=ulam/ys                                        1304
      call spelnet2(parm,ni,y,w,no,ne,nx,x,ix,jx,ju,vp,cl,nlam,flmin,vla   1306 
     *m,thr,maxit,  xm,xs,xv,lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                 1307
11680 do 11681 k=1,lmu                                                     1307
      alm(k)=ys*alm(k)                                                     1307
      nk=nin(k)                                                            1308
11690 do 11691 l=1,nk                                                      1308
      ca(l,k)=ys*ca(l,k)/xs(ia(l))                                         1308
11691 continue                                                             1308
11692 continue                                                             1308
      a0(k)=0.0                                                            1309
      if(intr.ne.0) a0(k)=ym-dot_product(ca(1:nk,k),xm(ia(1:nk)))          1310
11681 continue                                                             1311
11682 continue                                                             1311
      deallocate(xm,xs,ju,xv,vlam)                                         1312
      return                                                               1313
      end                                                                  1314
      subroutine spstandard1 (no,ni,x,ix,jx,y,w,ju,isd,intr,xm,xs,ym,ys,   1315 
     *xv,jerr)
      real x(*),y(no),w(no),xm(ni),xs(ni),xv(ni)                           1315
      integer ix(*),jx(*),ju(ni)                                           1316
      w=w/sum(w)                                                           1317
      if(intr .ne. 0)goto 11711                                            1317
      ym=0.0                                                               1318
      ys=sqrt(dot_product(w,y**2)-dot_product(w,y)**2)                     1318
      y=y/ys                                                               1319
11720 do 11721 j=1,ni                                                      1319
      if(ju(j).eq.0)goto 11721                                             1319
      xm(j)=0.0                                                            1319
      jb=ix(j)                                                             1319
      je=ix(j+1)-1                                                         1320
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)                          1321
      if(isd .eq. 0)goto 11741                                             1321
      xbq=dot_product(w(jx(jb:je)),x(jb:je))**2                            1321
      vc=xv(j)-xbq                                                         1322
      xs(j)=sqrt(vc)                                                       1322
      xv(j)=1.0+xbq/vc                                                     1323
      goto 11751                                                           1324
11741 continue                                                             1324
      xs(j)=1.0                                                            1324
11751 continue                                                             1325
11731 continue                                                             1325
11721 continue                                                             1326
11722 continue                                                             1326
      return                                                               1327
11711 continue                                                             1328
11760 do 11761 j=1,ni                                                      1328
      if(ju(j).eq.0)goto 11761                                             1329
      jb=ix(j)                                                             1329
      je=ix(j+1)-1                                                         1329
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             1330
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j)**2                 1331
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                       1332
11761 continue                                                             1333
11762 continue                                                             1333
      if(isd .ne. 0)goto 11781                                             1333
      xs=1.0                                                               1333
      goto 11791                                                           1333
11781 continue                                                             1333
      xv=1.0                                                               1333
11791 continue                                                             1334
11771 continue                                                             1334
      ym=dot_product(w,y)                                                  1334
      y=y-ym                                                               1334
      ys=sqrt(dot_product(w,y**2))                                         1334
      y=y/ys                                                               1335
      return                                                               1336
      end                                                                  1337
      subroutine spelnet2(beta,ni,y,w,no,ne,nx,x,ix,jx,ju,vp,cl,nlam,flm   1339 
     *in,ulam,  thr,maxit,xm,xs,xv,lmu,ao,ia,kin,rsqo,almo,nlp,jerr)
      real y(no),w(no),x(*),vp(ni),ulam(nlam),cl(2,ni)                     1340
      real ao(nx,nlam),rsqo(nlam),almo(nlam),xm(ni),xs(ni),xv(ni)          1341
      integer ix(*),jx(*),ju(ni),ia(nx),kin(nlam)                          1342
      real, dimension (:), allocatable :: a,g                                   
      integer, dimension (:), allocatable :: mm,iy                              
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)               1347
      allocate(a(1:ni),stat=jerr)                                          1348
      allocate(mm(1:ni),stat=ierr)                                         1348
      jerr=jerr+ierr                                                       1349
      allocate(g(1:ni),stat=ierr)                                          1349
      jerr=jerr+ierr                                                       1350
      allocate(iy(1:ni),stat=ierr)                                         1350
      jerr=jerr+ierr                                                       1351
      if(jerr.ne.0) return                                                 1352
      bta=beta                                                             1352
      omb=1.0-bta                                                          1352
      alm=0.0                                                              1352
      iy=0                                                                 1353
      if(flmin .ge. 1.0)goto 11811                                         1353
      eqs=max(eps,flmin)                                                   1353
      alf=eqs**(1.0/(nlam-1))                                              1353
11811 continue                                                             1354
      rsq=0.0                                                              1354
      a=0.0                                                                1354
      mm=0                                                                 1354
      o=0.0                                                                1354
      nlp=0                                                                1354
      nin=nlp                                                              1354
      iz=0                                                                 1354
      mnl=min(mnlam,nlam)                                                  1355
11820 do 11821 j=1,ni                                                      1355
      if(ju(j).eq.0)goto 11821                                             1356
      jb=ix(j)                                                             1356
      je=ix(j+1)-1                                                         1357
      g(j)=abs(dot_product(y(jx(jb:je))+o,w(jx(jb:je))*x(jb:je))/xs(j))    1358
11821 continue                                                             1359
11822 continue                                                             1359
11830 do 11831 m=1,nlam                                                    1359
      alm0=alm                                                             1360
      if(flmin .lt. 1.0)goto 11851                                         1360
      alm=ulam(m)                                                          1360
      goto 11841                                                           1361
11851 if(m .le. 2)goto 11861                                               1361
      alm=alm*alf                                                          1361
      goto 11841                                                           1362
11861 if(m .ne. 1)goto 11871                                               1362
      alm=big                                                              1362
      goto 11881                                                           1363
11871 continue                                                             1363
      alm0=0.0                                                             1364
11890 do 11891 j=1,ni                                                      1364
      if(ju(j).eq.0)goto 11891                                             1364
      if(vp(j).gt.0.0) alm0=max(alm0,g(j)/vp(j))                           1364
11891 continue                                                             1365
11892 continue                                                             1365
      alm0=alm0/max(bta,1.0e-3)                                            1365
      alm=alf*alm0                                                         1366
11881 continue                                                             1367
11841 continue                                                             1367
      dem=alm*omb                                                          1367
      ab=alm*bta                                                           1367
      rsq0=rsq                                                             1367
      jz=1                                                                 1368
      tlam=bta*(2.0*alm-alm0)                                              1369
11900 do 11901 k=1,ni                                                      1369
      if(iy(k).eq.1)goto 11901                                             1369
      if(ju(k).eq.0)goto 11901                                             1370
      if(g(k).gt.tlam*vp(k)) iy(k)=1                                       1371
11901 continue                                                             1372
11902 continue                                                             1372
11910 continue                                                             1372
11911 continue                                                             1372
      if(iz*jz.ne.0) go to 10360                                           1373
10880 continue                                                             1373
      nlp=nlp+1                                                            1373
      dlx=0.0                                                              1374
11920 do 11921 k=1,ni                                                      1374
      if(iy(k).eq.0)goto 11921                                             1374
      jb=ix(k)                                                             1374
      je=ix(k+1)-1                                                         1375
      gk=dot_product(y(jx(jb:je))+o,w(jx(jb:je))*x(jb:je))/xs(k)           1376
      ak=a(k)                                                              1376
      u=gk+ak*xv(k)                                                        1376
      v=abs(u)-vp(k)*ab                                                    1376
      a(k)=0.0                                                             1378
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1379 
     *em)))
      if(a(k).eq.ak)goto 11921                                             1380
      if(mm(k) .ne. 0)goto 11941                                           1380
      nin=nin+1                                                            1380
      if(nin.gt.nx)goto 11922                                              1381
      mm(k)=nin                                                            1381
      ia(nin)=k                                                            1382
11941 continue                                                             1383
      del=a(k)-ak                                                          1383
      rsq=rsq+del*(2.0*gk-del*xv(k))                                       1384
      y(jx(jb:je))=y(jx(jb:je))-del*x(jb:je)/xs(k)                         1385
      o=o+del*xm(k)/xs(k)                                                  1385
      dlx=max(xv(k)*del**2,dlx)                                            1386
11921 continue                                                             1387
11922 continue                                                             1387
      if(nin.gt.nx)goto 11912                                              1388
      if(dlx .ge. thr)goto 11961                                           1388
      ixx=0                                                                1389
11970 do 11971 j=1,ni                                                      1389
      if(iy(j).eq.1)goto 11971                                             1389
      if(ju(j).eq.0)goto 11971                                             1390
      jb=ix(j)                                                             1390
      je=ix(j+1)-1                                                         1391
      g(j)=abs(dot_product(y(jx(jb:je))+o,w(jx(jb:je))*x(jb:je))/xs(j))    1392
      if(g(j) .le. ab*vp(j))goto 11991                                     1392
      iy(j)=1                                                              1392
      ixx=1                                                                1392
11991 continue                                                             1393
11971 continue                                                             1394
11972 continue                                                             1394
      if(ixx.eq.1) go to 10880                                             1395
      goto 11912                                                           1396
11961 continue                                                             1397
      if(nlp .le. maxit)goto 12011                                         1397
      jerr=-m                                                              1397
      return                                                               1397
12011 continue                                                             1398
10360 continue                                                             1398
      iz=1                                                                 1399
12020 continue                                                             1399
12021 continue                                                             1399
      nlp=nlp+1                                                            1399
      dlx=0.0                                                              1400
12030 do 12031 l=1,nin                                                     1400
      k=ia(l)                                                              1400
      jb=ix(k)                                                             1400
      je=ix(k+1)-1                                                         1401
      gk=dot_product(y(jx(jb:je))+o,w(jx(jb:je))*x(jb:je))/xs(k)           1402
      ak=a(k)                                                              1402
      u=gk+ak*xv(k)                                                        1402
      v=abs(u)-vp(k)*ab                                                    1402
      a(k)=0.0                                                             1404
      if(v.gt.0.0) a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*d   1405 
     *em)))
      if(a(k).eq.ak)goto 12031                                             1406
      del=a(k)-ak                                                          1406
      rsq=rsq+del*(2.0*gk-del*xv(k))                                       1407
      y(jx(jb:je))=y(jx(jb:je))-del*x(jb:je)/xs(k)                         1408
      o=o+del*xm(k)/xs(k)                                                  1408
      dlx=max(xv(k)*del**2,dlx)                                            1409
12031 continue                                                             1410
12032 continue                                                             1410
      if(dlx.lt.thr)goto 12022                                             1410
      if(nlp .le. maxit)goto 12051                                         1410
      jerr=-m                                                              1410
      return                                                               1410
12051 continue                                                             1411
      goto 12021                                                           1412
12022 continue                                                             1412
      jz=0                                                                 1413
      goto 11911                                                           1414
11912 continue                                                             1414
      if(nin .le. nx)goto 12071                                            1414
      jerr=-10000-m                                                        1414
      goto 11832                                                           1414
12071 continue                                                             1415
      if(nin.gt.0) ao(1:nin,m)=a(ia(1:nin))                                1415
      kin(m)=nin                                                           1416
      rsqo(m)=rsq                                                          1416
      almo(m)=alm                                                          1416
      lmu=m                                                                1417
      if(m.lt.mnl)goto 11831                                               1417
      if(flmin.ge.1.0)goto 11831                                           1418
      me=0                                                                 1418
12080 do 12081 j=1,nin                                                     1418
      if(ao(j,m).ne.0.0) me=me+1                                           1418
12081 continue                                                             1418
12082 continue                                                             1418
      if(me.gt.ne)goto 11832                                               1419
      if(rsq-rsq0.lt.sml*rsq)goto 11832                                    1419
      if(rsq.gt.rsqmax)goto 11832                                          1420
11831 continue                                                             1421
11832 continue                                                             1421
      deallocate(a,mm,g,iy)                                                1422
      return                                                               1423
      end                                                                  1424
      subroutine spchkvars(no,ni,x,ix,ju)                                  1425
      real x(*)                                                            1425
      integer ix(*),ju(ni)                                                 1426
12090 do 12091 j=1,ni                                                      1426
      ju(j)=0                                                              1426
      jb=ix(j)                                                             1426
      nj=ix(j+1)-jb                                                        1426
      if(nj.eq.0)goto 12091                                                1427
      je=ix(j+1)-1                                                         1428
      if(nj .ge. no)goto 12111                                             1428
12120 do 12121 i=jb,je                                                     1428
      if(x(i).eq.0.0)goto 12121                                            1428
      ju(j)=1                                                              1428
      goto 12122                                                           1428
12121 continue                                                             1428
12122 continue                                                             1428
      goto 12131                                                           1429
12111 continue                                                             1429
      t=x(jb)                                                              1429
12140 do 12141 i=jb+1,je                                                   1429
      if(x(i).eq.t)goto 12141                                              1429
      ju(j)=1                                                              1429
      goto 12142                                                           1429
12141 continue                                                             1429
12142 continue                                                             1429
12131 continue                                                             1430
12101 continue                                                             1430
12091 continue                                                             1431
12092 continue                                                             1431
      return                                                               1432
      end                                                                  1433
      subroutine cmodval(a0,ca,ia,nin,x,ix,jx,n,f)                         1434
      real ca(*),x(*),f(n)                                                 1434
      integer ia(*),ix(*),jx(*)                                            1435
      f=a0                                                                 1436
12150 do 12151 j=1,nin                                                     1436
      k=ia(j)                                                              1436
      kb=ix(k)                                                             1436
      ke=ix(k+1)-1                                                         1437
      f(jx(kb:ke))=f(jx(kb:ke))+ca(j)*x(kb:ke)                             1438
12151 continue                                                             1439
12152 continue                                                             1439
      return                                                               1440
      end                                                                  1441
      function row_prod(i,j,ia,ja,ra,w)                                    1442
      integer ia(*),ja(*)                                                  1442
      real ra(*),w(*)                                                      1443
      row_prod=dot(ra(ia(i)),ra(ia(j)),ja(ia(i)),ja(ia(j)),  ia(i+1)-ia(   1445 
     *i),ia(j+1)-ia(j),w)
      return                                                               1446
      end                                                                  1447
      function dot(x,y,mx,my,nx,ny,w)                                      1448
      real x(*),y(*),w(*)                                                  1448
      integer mx(*),my(*)                                                  1449
      i=1                                                                  1449
      j=i                                                                  1449
      s=0.0                                                                1450
12160 continue                                                             1450
12161 continue                                                             1450
12170 continue                                                             1451
12171 if(mx(i).ge.my(j))goto 12172                                         1451
      i=i+1                                                                1451
      if(i.gt.nx) go to 12180                                              1451
      goto 12171                                                           1452
12172 continue                                                             1452
      if(mx(i).eq.my(j)) go to 12190                                       1453
12200 continue                                                             1453
12201 if(my(j).ge.mx(i))goto 12202                                         1453
      j=j+1                                                                1453
      if(j.gt.ny) go to 12180                                              1453
      goto 12201                                                           1454
12202 continue                                                             1454
      if(mx(i).eq.my(j)) go to 12190                                       1454
      goto 12161                                                           1455
12190 continue                                                             1455
      s=s+w(mx(i))*x(i)*y(j)                                               1456
      i=i+1                                                                1456
      if(i.gt.nx)goto 12162                                                1456
      j=j+1                                                                1456
      if(j.gt.ny)goto 12162                                                1457
      goto 12161                                                           1458
12162 continue                                                             1458
12180 continue                                                             1458
      dot=s                                                                1459
      return                                                               1460
      end                                                                  1461
      subroutine lognet (parm,no,ni,nc,x,y,g,jd,vp,cl,ne,nx,nlam,flmin,u   1463 
     *lam,thr,  isd,intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,je
     *rr)
      real x(no,ni),y(no,max(2,nc)),g(no,nc),vp(ni),ulam(nlam)             1464
      real ca(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),cl(2,ni)         1465
      integer jd(*),ia(nx),nin(nlam)                                       1466
      real, dimension (:), allocatable :: xm,xs,ww,vq,xv                        
      integer, dimension (:), allocatable :: ju                                 
      if(maxval(vp) .gt. 0.0)goto 12221                                    1470
      jerr=10000                                                           1470
      return                                                               1470
12221 continue                                                             1471
      allocate(ww(1:no),stat=jerr)                                         1472
      allocate(ju(1:ni),stat=ierr)                                         1472
      jerr=jerr+ierr                                                       1473
      allocate(vq(1:ni),stat=ierr)                                         1473
      jerr=jerr+ierr                                                       1474
      allocate(xm(1:ni),stat=ierr)                                         1474
      jerr=jerr+ierr                                                       1475
      if(kopt .ne. 2)goto 12241                                            1475
      allocate(xv(1:ni),stat=ierr)                                         1475
      jerr=jerr+ierr                                                       1475
12241 continue                                                             1476
      if(isd .le. 0)goto 12261                                             1476
      allocate(xs(1:ni),stat=ierr)                                         1476
      jerr=jerr+ierr                                                       1476
12261 continue                                                             1477
      if(jerr.ne.0) return                                                 1478
      call chkvars(no,ni,x,ju)                                             1479
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 1480
      if(maxval(ju) .gt. 0)goto 12281                                      1480
      jerr=7777                                                            1480
      return                                                               1480
12281 continue                                                             1481
      vq=max(0.0,vp)                                                       1481
      vq=vq*ni/sum(vq)                                                     1482
12290 do 12291 i=1,no                                                      1482
      ww(i)=sum(y(i,:))                                                    1482
      if(ww(i).gt.0.0) y(i,:)=y(i,:)/ww(i)                                 1482
12291 continue                                                             1483
12292 continue                                                             1483
      sw=sum(ww)                                                           1483
      ww=ww/sw                                                             1484
      if(nc .ne. 1)goto 12311                                              1484
      call lstandard1(no,ni,x,ww,ju,isd,intr,xm,xs)                        1485
      if(isd .le. 0)goto 12331                                             1485
12340 do 12341 j=1,ni                                                      1485
      cl(:,j)=cl(:,j)*xs(j)                                                1485
12341 continue                                                             1485
12342 continue                                                             1485
12331 continue                                                             1486
      call lognet2n(parm,no,ni,x,y(:,1),g(:,1),ww,ju,vq,cl,ne,nx,nlam,fl   1488 
     *min,ulam,  thr,isd,intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,dev,alm,n
     *lp,jerr)
      goto 12301                                                           1489
12311 if(kopt .ne. 2)goto 12351                                            1489
      call multlstandard1(no,ni,x,ww,ju,isd,intr,xm,xs,xv)                 1490
      if(isd .le. 0)goto 12371                                             1490
12380 do 12381 j=1,ni                                                      1490
      cl(:,j)=cl(:,j)*xs(j)                                                1490
12381 continue                                                             1490
12382 continue                                                             1490
12371 continue                                                             1491
      call multlognetn(parm,no,ni,nc,x,y,g,ww,ju,vq,cl,ne,nx,nlam,flmin,   1493 
     *ulam,thr,  intr,maxit,xv,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      goto 12391                                                           1494
12351 continue                                                             1494
      call lstandard1(no,ni,x,ww,ju,isd,intr,xm,xs)                        1495
      if(isd .le. 0)goto 12411                                             1495
12420 do 12421 j=1,ni                                                      1495
      cl(:,j)=cl(:,j)*xs(j)                                                1495
12421 continue                                                             1495
12422 continue                                                             1495
12411 continue                                                             1496
      call lognetn(parm,no,ni,nc,x,y,g,ww,ju,vq,cl,ne,nx,nlam,flmin,ulam   1498 
     *,thr,  isd,intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,jerr)
12391 continue                                                             1499
12301 continue                                                             1499
      if(jerr.gt.0) return                                                 1499
      dev0=2.0*sw*dev0                                                     1500
12430 do 12431 k=1,lmu                                                     1500
      nk=nin(k)                                                            1501
12440 do 12441 ic=1,nc                                                     1501
      if(isd .le. 0)goto 12461                                             1501
12470 do 12471 l=1,nk                                                      1501
      ca(l,ic,k)=ca(l,ic,k)/xs(ia(l))                                      1501
12471 continue                                                             1501
12472 continue                                                             1501
12461 continue                                                             1502
      if(intr .ne. 0)goto 12491                                            1502
      a0(ic,k)=0.0                                                         1502
      goto 12501                                                           1503
12491 continue                                                             1503
      a0(ic,k)=a0(ic,k)-dot_product(ca(1:nk,ic,k),xm(ia(1:nk)))            1503
12501 continue                                                             1504
12481 continue                                                             1504
12441 continue                                                             1505
12442 continue                                                             1505
12431 continue                                                             1506
12432 continue                                                             1506
      deallocate(ww,ju,vq,xm)                                              1506
      if(isd.gt.0) deallocate(xs)                                          1507
      if(kopt.eq.2) deallocate(xv)                                         1508
      return                                                               1509
      end                                                                  1510
      subroutine lstandard1 (no,ni,x,w,ju,isd,intr,xm,xs)                  1511
      real x(no,ni),w(no),xm(ni),xs(ni)                                    1511
      integer ju(ni)                                                       1512
      if(intr .ne. 0)goto 12521                                            1513
12530 do 12531 j=1,ni                                                      1513
      if(ju(j).eq.0)goto 12531                                             1513
      xm(j)=0.0                                                            1514
      if(isd .eq. 0)goto 12551                                             1514
      vc=dot_product(w,x(:,j)**2)-dot_product(w,x(:,j))**2                 1515
      xs(j)=sqrt(vc)                                                       1515
      x(:,j)=x(:,j)/xs(j)                                                  1516
12551 continue                                                             1517
12531 continue                                                             1518
12532 continue                                                             1518
      return                                                               1519
12521 continue                                                             1520
12560 do 12561 j=1,ni                                                      1520
      if(ju(j).eq.0)goto 12561                                             1521
      xm(j)=dot_product(w,x(:,j))                                          1521
      x(:,j)=x(:,j)-xm(j)                                                  1522
      if(isd .le. 0)goto 12581                                             1522
      xs(j)=sqrt(dot_product(w,x(:,j)**2))                                 1522
      x(:,j)=x(:,j)/xs(j)                                                  1522
12581 continue                                                             1523
12561 continue                                                             1524
12562 continue                                                             1524
      return                                                               1525
      end                                                                  1526
      subroutine multlstandard1 (no,ni,x,w,ju,isd,intr,xm,xs,xv)           1527
      real x(no,ni),w(no),xm(ni),xs(ni),xv(ni)                             1527
      integer ju(ni)                                                       1528
      if(intr .ne. 0)goto 12601                                            1529
12610 do 12611 j=1,ni                                                      1529
      if(ju(j).eq.0)goto 12611                                             1529
      xm(j)=0.0                                                            1530
      xv(j)=dot_product(w,x(:,j)**2)                                       1531
      if(isd .eq. 0)goto 12631                                             1531
      xbq=dot_product(w,x(:,j))**2                                         1531
      vc=xv(j)-xbq                                                         1532
      xs(j)=sqrt(vc)                                                       1532
      x(:,j)=x(:,j)/xs(j)                                                  1532
      xv(j)=1.0+xbq/vc                                                     1533
12631 continue                                                             1534
12611 continue                                                             1535
12612 continue                                                             1535
      return                                                               1536
12601 continue                                                             1537
12640 do 12641 j=1,ni                                                      1537
      if(ju(j).eq.0)goto 12641                                             1538
      xm(j)=dot_product(w,x(:,j))                                          1538
      x(:,j)=x(:,j)-xm(j)                                                  1539
      xv(j)=dot_product(w,x(:,j)**2)                                       1540
      if(isd .le. 0)goto 12661                                             1540
      xs(j)=sqrt(xv(j))                                                    1540
      x(:,j)=x(:,j)/xs(j)                                                  1540
      xv(j)=1.0                                                            1540
12661 continue                                                             1541
12641 continue                                                             1542
12642 continue                                                             1542
      return                                                               1543
      end                                                                  1544
      subroutine lognet2n(parm,no,ni,x,y,g,w,ju,vp,cl,ne,nx,nlam,flmin,u   1546 
     *lam,shri,  isd,intr,maxit,kopt,lmu,a0,a,m,kin,dev0,dev,alm,nlp,jer
     *r)
      real x(no,ni),y(no),g(no),w(no),vp(ni),ulam(nlam),cl(2,ni)           1547
      real a(nx,nlam),a0(nlam),dev(nlam),alm(nlam)                         1548
      integer ju(ni),m(nx),kin(nlam)                                       1549
      real, dimension (:), allocatable :: b,bs,v,r,xv,q,ga                      
      integer, dimension (:), allocatable :: mm,ixx                             
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               1554
      allocate(b(0:ni),stat=jerr)                                          1555
      allocate(xv(1:ni),stat=ierr)                                         1555
      jerr=jerr+ierr                                                       1556
      allocate(ga(1:ni),stat=ierr)                                         1556
      jerr=jerr+ierr                                                       1557
      allocate(bs(0:ni),stat=ierr)                                         1557
      jerr=jerr+ierr                                                       1558
      allocate(mm(1:ni),stat=ierr)                                         1558
      jerr=jerr+ierr                                                       1559
      allocate(ixx(1:ni),stat=ierr)                                        1559
      jerr=jerr+ierr                                                       1560
      allocate(r(1:no),stat=ierr)                                          1560
      jerr=jerr+ierr                                                       1561
      allocate(v(1:no),stat=ierr)                                          1561
      jerr=jerr+ierr                                                       1562
      allocate(q(1:no),stat=ierr)                                          1562
      jerr=jerr+ierr                                                       1563
      if(jerr.ne.0) return                                                 1564
      fmax=log(1.0/pmin-1.0)                                               1564
      fmin=-fmax                                                           1564
      vmin=(1.0+pmin)*pmin*(1.0-pmin)                                      1565
      bta=parm                                                             1565
      omb=1.0-bta                                                          1566
      q0=dot_product(w,y)                                                  1566
      if(q0 .gt. pmin)goto 12681                                           1566
      jerr=8001                                                            1566
      return                                                               1566
12681 continue                                                             1567
      if(q0 .lt. 1.0-pmin)goto 12701                                       1567
      jerr=9001                                                            1567
      return                                                               1567
12701 continue                                                             1568
      if(intr.eq.0.0) q0=0.5                                               1569
      ixx=0                                                                1569
      al=0.0                                                               1569
      bz=0.0                                                               1569
      if(intr.ne.0) bz=log(q0/(1.0-q0))                                    1570
      if(nonzero(no,g) .ne. 0)goto 12721                                   1570
      vi=q0*(1.0-q0)                                                       1570
      b(0)=bz                                                              1570
      v=vi*w                                                               1571
      r=w*(y-q0)                                                           1571
      q=q0                                                                 1571
      xmz=vi                                                               1571
      dev1=-(bz*q0+log(1.0-q0))                                            1572
      goto 12731                                                           1573
12721 continue                                                             1573
      b(0)=0.0                                                             1574
      if(intr .eq. 0)goto 12751                                            1574
      b(0)=azero(no,y,g,w,jerr)                                            1574
      if(jerr.ne.0) return                                                 1574
12751 continue                                                             1575
      q=1.0/(1.0+exp(-b(0)-g))                                             1575
      v=w*q*(1.0-q)                                                        1575
      r=w*(y-q)                                                            1575
      xmz=sum(v)                                                           1576
      dev1=-(b(0)*q0+dot_product(w,y*g+log(1.0-q)))                        1577
12731 continue                                                             1578
12711 continue                                                             1578
      if(kopt .le. 0)goto 12771                                            1579
      if(isd .le. 0 .or. intr .eq. 0)goto 12791                            1579
      xv=0.25                                                              1579
      goto 12801                                                           1580
12791 continue                                                             1580
12810 do 12811 j=1,ni                                                      1580
      if(ju(j).ne.0) xv(j)=0.25*dot_product(w,x(:,j)**2)                   1580
12811 continue                                                             1580
12812 continue                                                             1580
12801 continue                                                             1581
12781 continue                                                             1581
12771 continue                                                             1582
      dev0=dev1                                                            1583
12820 do 12821 i=1,no                                                      1583
      if(y(i).gt.0.0) dev0=dev0+w(i)*y(i)*log(y(i))                        1584
      if(y(i).lt.1.0) dev0=dev0+w(i)*(1.0-y(i))*log(1.0-y(i))              1585
12821 continue                                                             1586
12822 continue                                                             1586
      if(flmin .ge. 1.0)goto 12841                                         1586
      eqs=max(eps,flmin)                                                   1586
      alf=eqs**(1.0/(nlam-1))                                              1586
12841 continue                                                             1587
      m=0                                                                  1587
      mm=0                                                                 1587
      nlp=0                                                                1587
      nin=nlp                                                              1587
      mnl=min(mnlam,nlam)                                                  1587
      bs=0.0                                                               1587
      b(1:ni)=0.0                                                          1588
      shr=shri*dev0                                                        1589
12850 do 12851 j=1,ni                                                      1589
      if(ju(j).eq.0)goto 12851                                             1589
      ga(j)=abs(dot_product(r,x(:,j)))                                     1589
12851 continue                                                             1590
12852 continue                                                             1590
12860 do 12861 ilm=1,nlam                                                  1590
      al0=al                                                               1591
      if(flmin .lt. 1.0)goto 12881                                         1591
      al=ulam(ilm)                                                         1591
      goto 12871                                                           1592
12881 if(ilm .le. 2)goto 12891                                             1592
      al=al*alf                                                            1592
      goto 12871                                                           1593
12891 if(ilm .ne. 1)goto 12901                                             1593
      al=big                                                               1593
      goto 12911                                                           1594
12901 continue                                                             1594
      al0=0.0                                                              1595
12920 do 12921 j=1,ni                                                      1595
      if(ju(j).eq.0)goto 12921                                             1595
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            1595
12921 continue                                                             1596
12922 continue                                                             1596
      al0=al0/max(bta,1.0e-3)                                              1596
      al=alf*al0                                                           1597
12911 continue                                                             1598
12871 continue                                                             1598
      al2=al*omb                                                           1598
      al1=al*bta                                                           1598
      tlam=bta*(2.0*al-al0)                                                1599
12930 do 12931 k=1,ni                                                      1599
      if(ixx(k).eq.1)goto 12931                                            1599
      if(ju(k).eq.0)goto 12931                                             1600
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     1601
12931 continue                                                             1602
12932 continue                                                             1602
10880 continue                                                             1603
12940 continue                                                             1603
12941 continue                                                             1603
      bs(0)=b(0)                                                           1603
      if(nin.gt.0) bs(m(1:nin))=b(m(1:nin))                                1604
      if(kopt .ne. 0)goto 12961                                            1605
12970 do 12971 j=1,ni                                                      1605
      if(ixx(j).gt.0) xv(j)=dot_product(v,x(:,j)**2)                       1605
12971 continue                                                             1606
12972 continue                                                             1606
12961 continue                                                             1607
12980 continue                                                             1607
12981 continue                                                             1607
      nlp=nlp+1                                                            1607
      dlx=0.0                                                              1608
12990 do 12991 k=1,ni                                                      1608
      if(ixx(k).eq.0)goto 12991                                            1609
      bk=b(k)                                                              1609
      gk=dot_product(r,x(:,k))                                             1610
      u=gk+xv(k)*b(k)                                                      1610
      au=abs(u)-vp(k)*al1                                                  1611
      if(au .gt. 0.0)goto 13011                                            1611
      b(k)=0.0                                                             1611
      goto 13021                                                           1612
13011 continue                                                             1613
      b(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(xv(k)+vp(k)*al2)))          1614
13021 continue                                                             1615
13001 continue                                                             1615
      d=b(k)-bk                                                            1615
      if(abs(d).le.0.0)goto 12991                                          1615
      dlx=max(dlx,xv(k)*d**2)                                              1616
      r=r-d*v*x(:,k)                                                       1617
      if(mm(k) .ne. 0)goto 13041                                           1617
      nin=nin+1                                                            1617
      if(nin.gt.nx)goto 12992                                              1618
      mm(k)=nin                                                            1618
      m(nin)=k                                                             1619
13041 continue                                                             1620
12991 continue                                                             1621
12992 continue                                                             1621
      if(nin.gt.nx)goto 12982                                              1622
      d=0.0                                                                1622
      if(intr.ne.0) d=sum(r)/xmz                                           1623
      if(d .eq. 0.0)goto 13061                                             1623
      b(0)=b(0)+d                                                          1623
      dlx=max(dlx,xmz*d**2)                                                1623
      r=r-d*v                                                              1623
13061 continue                                                             1624
      if(dlx.lt.shr)goto 12982                                             1624
      if(nlp .le. maxit)goto 13081                                         1624
      jerr=-ilm                                                            1624
      return                                                               1624
13081 continue                                                             1625
13090 continue                                                             1625
13091 continue                                                             1625
      nlp=nlp+1                                                            1625
      dlx=0.0                                                              1626
13100 do 13101 l=1,nin                                                     1626
      k=m(l)                                                               1626
      bk=b(k)                                                              1627
      gk=dot_product(r,x(:,k))                                             1628
      u=gk+xv(k)*b(k)                                                      1628
      au=abs(u)-vp(k)*al1                                                  1629
      if(au .gt. 0.0)goto 13121                                            1629
      b(k)=0.0                                                             1629
      goto 13131                                                           1630
13121 continue                                                             1631
      b(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(xv(k)+vp(k)*al2)))          1632
13131 continue                                                             1633
13111 continue                                                             1633
      d=b(k)-bk                                                            1633
      if(abs(d).le.0.0)goto 13101                                          1633
      dlx=max(dlx,xv(k)*d**2)                                              1634
      r=r-d*v*x(:,k)                                                       1635
13101 continue                                                             1636
13102 continue                                                             1636
      d=0.0                                                                1636
      if(intr.ne.0) d=sum(r)/xmz                                           1637
      if(d .eq. 0.0)goto 13151                                             1637
      b(0)=b(0)+d                                                          1637
      dlx=max(dlx,xmz*d**2)                                                1637
      r=r-d*v                                                              1637
13151 continue                                                             1638
      if(dlx.lt.shr)goto 13092                                             1638
      if(nlp .le. maxit)goto 13171                                         1638
      jerr=-ilm                                                            1638
      return                                                               1638
13171 continue                                                             1639
      goto 13091                                                           1640
13092 continue                                                             1640
      goto 12981                                                           1641
12982 continue                                                             1641
      if(nin.gt.nx)goto 12942                                              1642
13180 do 13181 i=1,no                                                      1642
      fi=b(0)+g(i)                                                         1643
      if(nin.gt.0) fi=fi+dot_product(b(m(1:nin)),x(i,m(1:nin)))            1644
      if(fi .ge. fmin)goto 13201                                           1644
      q(i)=0.0                                                             1644
      goto 13191                                                           1644
13201 if(fi .le. fmax)goto 13211                                           1644
      q(i)=1.0                                                             1644
      goto 13221                                                           1645
13211 continue                                                             1645
      q(i)=1.0/(1.0+exp(-fi))                                              1645
13221 continue                                                             1646
13191 continue                                                             1646
13181 continue                                                             1647
13182 continue                                                             1647
      v=w*q*(1.0-q)                                                        1647
      xmz=sum(v)                                                           1647
      if(xmz.le.vmin)goto 12942                                            1647
      r=w*(y-q)                                                            1648
      if(xmz*(b(0)-bs(0))**2 .ge. shr)goto 13241                           1648
      ix=0                                                                 1649
13250 do 13251 j=1,nin                                                     1649
      k=m(j)                                                               1650
      if(xv(k)*(b(k)-bs(k))**2.lt.shr)goto 13251                           1650
      ix=1                                                                 1650
      goto 13252                                                           1651
13251 continue                                                             1652
13252 continue                                                             1652
      if(ix .ne. 0)goto 13271                                              1653
13280 do 13281 k=1,ni                                                      1653
      if(ixx(k).eq.1)goto 13281                                            1653
      if(ju(k).eq.0)goto 13281                                             1654
      ga(k)=abs(dot_product(r,x(:,k)))                                     1655
      if(ga(k) .le. al1*vp(k))goto 13301                                   1655
      ixx(k)=1                                                             1655
      ix=1                                                                 1655
13301 continue                                                             1656
13281 continue                                                             1657
13282 continue                                                             1657
      if(ix.eq.1) go to 10880                                              1658
      goto 12942                                                           1659
13271 continue                                                             1660
13241 continue                                                             1661
      goto 12941                                                           1662
12942 continue                                                             1662
      if(nin .le. nx)goto 13321                                            1662
      jerr=-10000-ilm                                                      1662
      goto 12862                                                           1662
13321 continue                                                             1663
      if(nin.gt.0) a(1:nin,ilm)=b(m(1:nin))                                1663
      kin(ilm)=nin                                                         1664
      a0(ilm)=b(0)                                                         1664
      alm(ilm)=al                                                          1664
      lmu=ilm                                                              1665
      devi=dev2(no,w,y,q,pmin)                                             1666
      dev(ilm)=(dev1-devi)/dev0                                            1666
      if(xmz.le.vmin)goto 12862                                            1667
      if(ilm.lt.mnl)goto 12861                                             1667
      if(flmin.ge.1.0)goto 12861                                           1668
      me=0                                                                 1668
13330 do 13331 j=1,nin                                                     1668
      if(a(j,ilm).ne.0.0) me=me+1                                          1668
13331 continue                                                             1668
13332 continue                                                             1668
      if(me.gt.ne)goto 12862                                               1669
      if(dev(ilm).gt.devmax)goto 12862                                     1669
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 12862                             1670
12861 continue                                                             1671
12862 continue                                                             1671
      g=log(q/(1.0-q))                                                     1672
      deallocate(b,bs,v,r,xv,q,mm,ga,ixx)                                  1673
      return                                                               1674
      end                                                                  1675
      function dev2(n,w,y,p,pmin)                                          1676
      real w(n),y(n),p(n)                                                  1677
      pmax=1.0-pmin                                                        1677
      s=0.0                                                                1678
13340 do 13341 i=1,n                                                       1678
      pi=min(max(pmin,p(i)),pmax)                                          1679
      s=s-w(i)*(y(i)*log(pi)+(1.0-y(i))*log(1.0-pi))                       1680
13341 continue                                                             1681
13342 continue                                                             1681
      dev2=s                                                               1682
      return                                                               1683
      end                                                                  1684
      function azero(n,y,g,q,jerr)                                         1685
      parameter(eps=1.0e-7)                                                1686
      real y(n),g(n),q(n)                                                  1687
      real, dimension (:), allocatable :: e,p,w                                 
      allocate(e(1:n),stat=jerr)                                           1691
      allocate(p(1:n),stat=ierr)                                           1691
      jerr=jerr+ierr                                                       1692
      allocate(w(1:n),stat=ierr)                                           1692
      jerr=jerr+ierr                                                       1693
      az=0.0                                                               1694
      azero=0.0                                                            1694
      if(jerr.ne.0) return                                                 1694
      e=exp(-g)                                                            1694
      qy=dot_product(q,y)                                                  1694
      p=1.0/(1.0+e)                                                        1695
13350 continue                                                             1695
13351 continue                                                             1695
      w=q*p*(1.0-p)                                                        1696
      d=(qy-dot_product(q,p))/sum(w)                                       1696
      az=az+d                                                              1696
      if(abs(d).lt.eps)goto 13352                                          1697
      ea0=exp(-az)                                                         1697
      p=1.0/(1.0+ea0*e)                                                    1698
      goto 13351                                                           1699
13352 continue                                                             1699
      azero=az                                                             1700
      deallocate(e,p,w)                                                    1701
      return                                                               1702
      end                                                                  1703
      subroutine lognetn(parm,no,ni,nc,x,y,g,w,ju,vp,cl,ne,nx,nlam,flmin   1705 
     *,ulam,shri,  isd,intr,maxit,kopt,lmu,a0,a,m,kin,dev0,dev,alm,nlp,j
     *err)
      real x(no,ni),y(no,nc),g(no,nc),w(no),vp(ni),ulam(nlam)              1706
      real a(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),cl(2,ni)          1707
      integer ju(ni),m(nx),kin(nlam)                                       1708
      real, dimension (:,:), allocatable :: q                                   
      real, dimension (:), allocatable :: sxp,sxpl                              
      real, dimension (:), allocatable :: di,v,r,ga                             
      real, dimension (:,:), allocatable :: b,bs,xv                             
      integer, dimension (:), allocatable :: mm,is,ixx                          
      allocate(b(0:ni,1:nc),stat=jerr)                                          
      allocate(xv(1:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(bs(0:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(q(1:no,1:nc),stat=ierr); jerr=jerr+ierr                          
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               1719
      exmn=-exmx                                                           1720
      allocate(r(1:no),stat=ierr)                                          1720
      jerr=jerr+ierr                                                       1721
      allocate(v(1:no),stat=ierr)                                          1721
      jerr=jerr+ierr                                                       1722
      allocate(mm(1:ni),stat=ierr)                                         1722
      jerr=jerr+ierr                                                       1723
      allocate(is(1:max(nc,ni)),stat=ierr)                                 1723
      jerr=jerr+ierr                                                       1724
      allocate(sxp(1:no),stat=ierr)                                        1724
      jerr=jerr+ierr                                                       1725
      allocate(sxpl(1:no),stat=ierr)                                       1725
      jerr=jerr+ierr                                                       1726
      allocate(di(1:no),stat=ierr)                                         1726
      jerr=jerr+ierr                                                       1727
      allocate(ga(1:ni),stat=ierr)                                         1727
      jerr=jerr+ierr                                                       1728
      allocate(ixx(1:ni),stat=ierr)                                        1728
      jerr=jerr+ierr                                                       1729
      if(jerr.ne.0) return                                                 1730
      pmax=1.0-pmin                                                        1730
      emin=pmin/pmax                                                       1730
      emax=1.0/emin                                                        1731
      pfm=(1.0+pmin)*pmin                                                  1731
      pfx=(1.0-pmin)*pmax                                                  1731
      vmin=pfm*pmax                                                        1732
      bta=parm                                                             1732
      omb=1.0-bta                                                          1732
      dev1=0.0                                                             1732
      dev0=0.0                                                             1733
13360 do 13361 ic=1,nc                                                     1733
      q0=dot_product(w,y(:,ic))                                            1734
      if(q0 .gt. pmin)goto 13381                                           1734
      jerr =8000+ic                                                        1734
      return                                                               1734
13381 continue                                                             1735
      if(q0 .lt. 1.0-pmin)goto 13401                                       1735
      jerr =9000+ic                                                        1735
      return                                                               1735
13401 continue                                                             1736
      if(intr .ne. 0)goto 13421                                            1736
      q0=1.0/nc                                                            1736
      b(0,ic)=0.0                                                          1736
      goto 13431                                                           1737
13421 continue                                                             1737
      b(0,ic)=log(q0)                                                      1737
      dev1=dev1-q0*b(0,ic)                                                 1737
13431 continue                                                             1738
13411 continue                                                             1738
      b(1:ni,ic)=0.0                                                       1739
13361 continue                                                             1740
13362 continue                                                             1740
      if(intr.eq.0) dev1=log(float(nc))                                    1740
      ixx=0                                                                1740
      al=0.0                                                               1741
      if(nonzero(no*nc,g) .ne. 0)goto 13451                                1742
      b(0,:)=b(0,:)-sum(b(0,:))/nc                                         1742
      sxp=0.0                                                              1743
13460 do 13461 ic=1,nc                                                     1743
      q(:,ic)=exp(b(0,ic))                                                 1743
      sxp=sxp+q(:,ic)                                                      1743
13461 continue                                                             1744
13462 continue                                                             1744
      goto 13471                                                           1745
13451 continue                                                             1745
13480 do 13481 i=1,no                                                      1745
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         1745
13481 continue                                                             1745
13482 continue                                                             1745
      sxp=0.0                                                              1746
      if(intr .ne. 0)goto 13501                                            1746
      b(0,:)=0.0                                                           1746
      goto 13511                                                           1747
13501 continue                                                             1747
      call kazero(nc,no,y,g,w,b(0,:),jerr)                                 1747
      if(jerr.ne.0) return                                                 1747
13511 continue                                                             1748
13491 continue                                                             1748
      dev1=0.0                                                             1749
13520 do 13521 ic=1,nc                                                     1749
      q(:,ic)=b(0,ic)+g(:,ic)                                              1750
      dev1=dev1-dot_product(w,y(:,ic)*q(:,ic))                             1751
      q(:,ic)=exp(q(:,ic))                                                 1751
      sxp=sxp+q(:,ic)                                                      1752
13521 continue                                                             1753
13522 continue                                                             1753
      sxpl=w*log(sxp)                                                      1753
13530 do 13531 ic=1,nc                                                     1753
      dev1=dev1+dot_product(y(:,ic),sxpl)                                  1753
13531 continue                                                             1754
13532 continue                                                             1754
13471 continue                                                             1755
13441 continue                                                             1755
13540 do 13541 ic=1,nc                                                     1755
13550 do 13551 i=1,no                                                      1755
      if(y(i,ic).gt.0.0) dev0=dev0+w(i)*y(i,ic)*log(y(i,ic))               1755
13551 continue                                                             1755
13552 continue                                                             1755
13541 continue                                                             1756
13542 continue                                                             1756
      dev0=dev0+dev1                                                       1757
      if(kopt .le. 0)goto 13571                                            1758
      if(isd .le. 0 .or. intr .eq. 0)goto 13591                            1758
      xv=0.25                                                              1758
      goto 13601                                                           1759
13591 continue                                                             1759
13610 do 13611 j=1,ni                                                      1759
      if(ju(j).ne.0) xv(j,:)=0.25*dot_product(w,x(:,j)**2)                 1759
13611 continue                                                             1759
13612 continue                                                             1759
13601 continue                                                             1760
13581 continue                                                             1760
13571 continue                                                             1761
      if(flmin .ge. 1.0)goto 13631                                         1761
      eqs=max(eps,flmin)                                                   1761
      alf=eqs**(1.0/(nlam-1))                                              1761
13631 continue                                                             1762
      m=0                                                                  1762
      mm=0                                                                 1762
      nin=0                                                                1762
      nlp=0                                                                1762
      mnl=min(mnlam,nlam)                                                  1762
      bs=0.0                                                               1762
      shr=shri*dev0                                                        1763
      ga=0.0                                                               1764
13640 do 13641 ic=1,nc                                                     1764
      r=w*(y(:,ic)-q(:,ic)/sxp)                                            1765
13650 do 13651 j=1,ni                                                      1765
      if(ju(j).ne.0) ga(j)=max(ga(j),abs(dot_product(r,x(:,j))))           1765
13651 continue                                                             1766
13652 continue                                                             1766
13641 continue                                                             1767
13642 continue                                                             1767
13660 do 13661 ilm=1,nlam                                                  1767
      al0=al                                                               1768
      if(flmin .lt. 1.0)goto 13681                                         1768
      al=ulam(ilm)                                                         1768
      goto 13671                                                           1769
13681 if(ilm .le. 2)goto 13691                                             1769
      al=al*alf                                                            1769
      goto 13671                                                           1770
13691 if(ilm .ne. 1)goto 13701                                             1770
      al=big                                                               1770
      goto 13711                                                           1771
13701 continue                                                             1771
      al0=0.0                                                              1772
13720 do 13721 j=1,ni                                                      1772
      if(ju(j).eq.0)goto 13721                                             1772
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            1772
13721 continue                                                             1773
13722 continue                                                             1773
      al0=al0/max(bta,1.0e-3)                                              1773
      al=alf*al0                                                           1774
13711 continue                                                             1775
13671 continue                                                             1775
      al2=al*omb                                                           1775
      al1=al*bta                                                           1775
      tlam=bta*(2.0*al-al0)                                                1776
13730 do 13731 k=1,ni                                                      1776
      if(ixx(k).eq.1)goto 13731                                            1776
      if(ju(k).eq.0)goto 13731                                             1777
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     1778
13731 continue                                                             1779
13732 continue                                                             1779
10880 continue                                                             1780
13740 continue                                                             1780
13741 continue                                                             1780
      ix=0                                                                 1780
      jx=ix                                                                1780
      ig=0                                                                 1781
13750 do 13751 ic=1,nc                                                     1781
      bs(0,ic)=b(0,ic)                                                     1782
      if(nin.gt.0) bs(m(1:nin),ic)=b(m(1:nin),ic)                          1783
      xmz=0.0                                                              1784
13760 do 13761 i=1,no                                                      1784
      pic=q(i,ic)/sxp(i)                                                   1785
      if(pic .ge. pfm)goto 13781                                           1785
      pic=0.0                                                              1785
      v(i)=0.0                                                             1785
      goto 13771                                                           1786
13781 if(pic .le. pfx)goto 13791                                           1786
      pic=1.0                                                              1786
      v(i)=0.0                                                             1786
      goto 13801                                                           1787
13791 continue                                                             1787
      v(i)=w(i)*pic*(1.0-pic)                                              1787
      xmz=xmz+v(i)                                                         1787
13801 continue                                                             1788
13771 continue                                                             1788
      r(i)=w(i)*(y(i,ic)-pic)                                              1789
13761 continue                                                             1790
13762 continue                                                             1790
      if(xmz.le.vmin)goto 13751                                            1790
      ig=1                                                                 1791
      if(kopt .ne. 0)goto 13821                                            1792
13830 do 13831 j=1,ni                                                      1792
      if(ixx(j).gt.0) xv(j,ic)=dot_product(v,x(:,j)**2)                    1792
13831 continue                                                             1793
13832 continue                                                             1793
13821 continue                                                             1794
13840 continue                                                             1794
13841 continue                                                             1794
      nlp=nlp+1                                                            1794
      dlx=0.0                                                              1795
13850 do 13851 k=1,ni                                                      1795
      if(ixx(k).eq.0)goto 13851                                            1796
      bk=b(k,ic)                                                           1796
      gk=dot_product(r,x(:,k))                                             1797
      u=gk+xv(k,ic)*b(k,ic)                                                1797
      au=abs(u)-vp(k)*al1                                                  1798
      if(au .gt. 0.0)goto 13871                                            1798
      b(k,ic)=0.0                                                          1798
      goto 13881                                                           1799
13871 continue                                                             1800
      b(k,ic)=max(cl(1,k),min(cl(2,k),sign(au,u)/  (xv(k,ic)+vp(k)*al2))   1802 
     *)
13881 continue                                                             1803
13861 continue                                                             1803
      d=b(k,ic)-bk                                                         1803
      if(abs(d).le.0.0)goto 13851                                          1804
      dlx=max(dlx,xv(k,ic)*d**2)                                           1804
      r=r-d*v*x(:,k)                                                       1805
      if(mm(k) .ne. 0)goto 13901                                           1805
      nin=nin+1                                                            1806
      if(nin .le. nx)goto 13921                                            1806
      jx=1                                                                 1806
      goto 13852                                                           1806
13921 continue                                                             1807
      mm(k)=nin                                                            1807
      m(nin)=k                                                             1808
13901 continue                                                             1809
13851 continue                                                             1810
13852 continue                                                             1810
      if(jx.gt.0)goto 13842                                                1811
      d=0.0                                                                1811
      if(intr.ne.0) d=sum(r)/xmz                                           1812
      if(d .eq. 0.0)goto 13941                                             1812
      b(0,ic)=b(0,ic)+d                                                    1812
      dlx=max(dlx,xmz*d**2)                                                1812
      r=r-d*v                                                              1812
13941 continue                                                             1813
      if(dlx.lt.shr)goto 13842                                             1814
      if(nlp .le. maxit)goto 13961                                         1814
      jerr=-ilm                                                            1814
      return                                                               1814
13961 continue                                                             1815
13970 continue                                                             1815
13971 continue                                                             1815
      nlp=nlp+1                                                            1815
      dlx=0.0                                                              1816
13980 do 13981 l=1,nin                                                     1816
      k=m(l)                                                               1816
      bk=b(k,ic)                                                           1817
      gk=dot_product(r,x(:,k))                                             1818
      u=gk+xv(k,ic)*b(k,ic)                                                1818
      au=abs(u)-vp(k)*al1                                                  1819
      if(au .gt. 0.0)goto 14001                                            1819
      b(k,ic)=0.0                                                          1819
      goto 14011                                                           1820
14001 continue                                                             1821
      b(k,ic)=max(cl(1,k),min(cl(2,k),sign(au,u)/  (xv(k,ic)+vp(k)*al2))   1823 
     *)
14011 continue                                                             1824
13991 continue                                                             1824
      d=b(k,ic)-bk                                                         1824
      if(abs(d).le.0.0)goto 13981                                          1825
      dlx=max(dlx,xv(k,ic)*d**2)                                           1825
      r=r-d*v*x(:,k)                                                       1826
13981 continue                                                             1827
13982 continue                                                             1827
      d=0.0                                                                1827
      if(intr.ne.0) d=sum(r)/xmz                                           1828
      if(d .eq. 0.0)goto 14031                                             1828
      b(0,ic)=b(0,ic)+d                                                    1829
      dlx=max(dlx,xmz*d**2)                                                1829
      r=r-d*v                                                              1830
14031 continue                                                             1831
      if(dlx.lt.shr)goto 13972                                             1831
      if(nlp .le. maxit)goto 14051                                         1831
      jerr=-ilm                                                            1831
      return                                                               1831
14051 continue                                                             1832
      goto 13971                                                           1833
13972 continue                                                             1833
      goto 13841                                                           1834
13842 continue                                                             1834
      if(jx.gt.0)goto 13752                                                1835
      if(xmz*(b(0,ic)-bs(0,ic))**2.gt.shr) ix=1                            1836
      if(ix .ne. 0)goto 14071                                              1837
14080 do 14081 j=1,nin                                                     1837
      k=m(j)                                                               1838
      if(xv(k,ic)*(b(k,ic)-bs(k,ic))**2 .le. shr)goto 14101                1838
      ix=1                                                                 1838
      goto 14082                                                           1838
14101 continue                                                             1839
14081 continue                                                             1840
14082 continue                                                             1840
14071 continue                                                             1841
14110 do 14111 i=1,no                                                      1841
      fi=b(0,ic)+g(i,ic)                                                   1843
      if(nin.gt.0) fi=fi+dot_product(b(m(1:nin),ic),x(i,m(1:nin)))         1844
      fi=min(max(exmn,fi),exmx)                                            1844
      sxp(i)=sxp(i)-q(i,ic)                                                1845
      q(i,ic)=min(max(emin*sxp(i),exp(fi)),emax*sxp(i))                    1846
      sxp(i)=sxp(i)+q(i,ic)                                                1847
14111 continue                                                             1848
14112 continue                                                             1848
13751 continue                                                             1849
13752 continue                                                             1849
      s=-sum(b(0,:))/nc                                                    1849
      b(0,:)=b(0,:)+s                                                      1849
      di=s                                                                 1850
14120 do 14121 j=1,nin                                                     1850
      l=m(j)                                                               1851
      if(vp(l) .gt. 0.0)goto 14141                                         1851
      s=sum(b(l,:))/nc                                                     1851
      goto 14151                                                           1852
14141 continue                                                             1852
      s=elc(parm,nc,cl(:,l),b(l,:),is)                                     1852
14151 continue                                                             1853
14131 continue                                                             1853
      b(l,:)=b(l,:)-s                                                      1853
      di=di-s*x(:,l)                                                       1854
14121 continue                                                             1855
14122 continue                                                             1855
      di=exp(di)                                                           1855
      sxp=sxp*di                                                           1855
14160 do 14161 ic=1,nc                                                     1855
      q(:,ic)=q(:,ic)*di                                                   1855
14161 continue                                                             1856
14162 continue                                                             1856
      if(jx.gt.0)goto 13742                                                1856
      if(ig.eq.0)goto 13742                                                1857
      if(ix .ne. 0)goto 14181                                              1858
14190 do 14191 k=1,ni                                                      1858
      if(ixx(k).eq.1)goto 14191                                            1858
      if(ju(k).eq.0)goto 14191                                             1858
      ga(k)=0.0                                                            1858
14191 continue                                                             1859
14192 continue                                                             1859
14200 do 14201 ic=1,nc                                                     1859
      r=w*(y(:,ic)-q(:,ic)/sxp)                                            1860
14210 do 14211 k=1,ni                                                      1860
      if(ixx(k).eq.1)goto 14211                                            1860
      if(ju(k).eq.0)goto 14211                                             1861
      ga(k)=max(ga(k),abs(dot_product(r,x(:,k))))                          1862
14211 continue                                                             1863
14212 continue                                                             1863
14201 continue                                                             1864
14202 continue                                                             1864
14220 do 14221 k=1,ni                                                      1864
      if(ixx(k).eq.1)goto 14221                                            1864
      if(ju(k).eq.0)goto 14221                                             1865
      if(ga(k) .le. al1*vp(k))goto 14241                                   1865
      ixx(k)=1                                                             1865
      ix=1                                                                 1865
14241 continue                                                             1866
14221 continue                                                             1867
14222 continue                                                             1867
      if(ix.eq.1) go to 10880                                              1868
      goto 13742                                                           1869
14181 continue                                                             1870
      goto 13741                                                           1871
13742 continue                                                             1871
      if(jx .le. 0)goto 14261                                              1871
      jerr=-10000-ilm                                                      1871
      goto 13662                                                           1871
14261 continue                                                             1871
      devi=0.0                                                             1872
14270 do 14271 ic=1,nc                                                     1873
      if(nin.gt.0) a(1:nin,ic,ilm)=b(m(1:nin),ic)                          1873
      a0(ic,ilm)=b(0,ic)                                                   1874
14280 do 14281 i=1,no                                                      1874
      if(y(i,ic).le.0.0)goto 14281                                         1875
      devi=devi-w(i)*y(i,ic)*log(q(i,ic)/sxp(i))                           1876
14281 continue                                                             1877
14282 continue                                                             1877
14271 continue                                                             1878
14272 continue                                                             1878
      kin(ilm)=nin                                                         1878
      alm(ilm)=al                                                          1878
      lmu=ilm                                                              1879
      dev(ilm)=(dev1-devi)/dev0                                            1879
      if(ig.eq.0)goto 13662                                                1880
      if(ilm.lt.mnl)goto 13661                                             1880
      if(flmin.ge.1.0)goto 13661                                           1881
      if(nintot(ni,nx,nc,a(1,1,ilm),m,nin,is).gt.ne)goto 13662             1882
      if(dev(ilm).gt.devmax)goto 13662                                     1882
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 13662                             1883
13661 continue                                                             1884
13662 continue                                                             1884
      g=log(q)                                                             1884
14290 do 14291 i=1,no                                                      1884
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         1884
14291 continue                                                             1885
14292 continue                                                             1885
      deallocate(sxp,b,bs,v,r,xv,q,mm,is,ga,ixx)                           1886
      return                                                               1887
      end                                                                  1888
      subroutine kazero(kk,n,y,g,q,az,jerr)                                1889
      parameter(eps=1.0e-7)                                                1890
      real y(n,kk),g(n,kk),q(n),az(kk)                                     1891
      real, dimension (:), allocatable :: s                                     
      real, dimension (:,:), allocatable :: e                                   
      allocate(e(1:n,1:kk),stat=jerr)                                           
      allocate(s(1:n),stat=ierr)                                           1896
      jerr=jerr+ierr                                                       1897
      if(jerr.ne.0) return                                                 1898
      az=0.0                                                               1898
      e=exp(g)                                                             1898
14300 do 14301 i=1,n                                                       1898
      s(i)=sum(e(i,:))                                                     1898
14301 continue                                                             1899
14302 continue                                                             1899
14310 continue                                                             1899
14311 continue                                                             1899
      dm=0.0                                                               1900
14320 do 14321 k=1,kk                                                      1900
      t=0.0                                                                1900
      u=t                                                                  1901
14330 do 14331 i=1,n                                                       1901
      pik=e(i,k)/s(i)                                                      1902
      t=t+q(i)*(y(i,k)-pik)                                                1902
      u=u+q(i)*pik*(1.0-pik)                                               1903
14331 continue                                                             1904
14332 continue                                                             1904
      d=t/u                                                                1904
      az(k)=az(k)+d                                                        1904
      ed=exp(d)                                                            1904
      dm=max(dm,abs(d))                                                    1905
14340 do 14341 i=1,n                                                       1905
      z=e(i,k)                                                             1905
      e(i,k)=z*ed                                                          1905
      s(i)=s(i)-z+e(i,k)                                                   1905
14341 continue                                                             1906
14342 continue                                                             1906
14321 continue                                                             1907
14322 continue                                                             1907
      if(dm.lt.eps)goto 14312                                              1907
      goto 14311                                                           1908
14312 continue                                                             1908
      az=az-sum(az)/kk                                                     1909
      deallocate(e,s)                                                      1910
      return                                                               1911
      end                                                                  1912
      function elc(parm,n,cl,a,m)                                          1913
      real a(n),cl(2)                                                      1913
      integer m(n)                                                         1914
      fn=n                                                                 1914
      am=sum(a)/fn                                                         1915
      if((parm .ne. 0.0) .and. (n .ne. 2))goto 14361                       1915
      elc=am                                                               1915
      go to 14370                                                          1915
14361 continue                                                             1916
14380 do 14381 i=1,n                                                       1916
      m(i)=i                                                               1916
14381 continue                                                             1916
14382 continue                                                             1916
      call psort7(a,m,1,n)                                                 1917
      if(a(m(1)) .ne. a(m(n)))goto 14401                                   1917
      elc=a(1)                                                             1917
      go to 14370                                                          1917
14401 continue                                                             1918
      if(mod(n,2) .ne. 1)goto 14421                                        1918
      ad=a(m(n/2+1))                                                       1918
      goto 14431                                                           1919
14421 continue                                                             1919
      ad=0.5*(a(m(n/2+1))+a(m(n/2)))                                       1919
14431 continue                                                             1920
14411 continue                                                             1920
      if(parm .ne. 1.0)goto 14451                                          1920
      elc=ad                                                               1920
      go to 14370                                                          1920
14451 continue                                                             1921
      b1=min(am,ad)                                                        1921
      b2=max(am,ad)                                                        1921
      k2=1                                                                 1922
14460 continue                                                             1922
14461 if(a(m(k2)).gt.b1)goto 14462                                         1922
      k2=k2+1                                                              1922
      goto 14461                                                           1922
14462 continue                                                             1922
      k1=k2-1                                                              1923
14470 continue                                                             1923
14471 if(a(m(k2)).ge.b2)goto 14472                                         1923
      k2=k2+1                                                              1923
      goto 14471                                                           1924
14472 continue                                                             1924
      r=parm/((1.0-parm)*fn)                                               1924
      is=0                                                                 1924
      sm=n-2*(k1-1)                                                        1925
14480 do 14481 k=k1,k2-1                                                   1925
      sm=sm-2.0                                                            1925
      s=r*sm+am                                                            1926
      if(s .le. a(m(k)) .or. s .gt. a(m(k+1)))goto 14501                   1926
      is=k                                                                 1926
      goto 14482                                                           1926
14501 continue                                                             1927
14481 continue                                                             1928
14482 continue                                                             1928
      if(is .eq. 0)goto 14521                                              1928
      elc=s                                                                1928
      go to 14370                                                          1928
14521 continue                                                             1928
      r2=2.0*r                                                             1928
      s1=a(m(k1))                                                          1928
      am2=2.0*am                                                           1929
      cri=r2*sum(abs(a-s1))+s1*(s1-am2)                                    1929
      elc=s1                                                               1930
14530 do 14531 k=k1+1,k2                                                   1930
      s=a(m(k))                                                            1930
      if(s.eq.s1)goto 14531                                                1931
      c=r2*sum(abs(a-s))+s*(s-am2)                                         1932
      if(c .ge. cri)goto 14551                                             1932
      cri=c                                                                1932
      elc=s                                                                1932
14551 continue                                                             1932
      s1=s                                                                 1933
14531 continue                                                             1934
14532 continue                                                             1934
14370 continue                                                             1934
      elc=max(maxval(a-cl(2)),min(minval(a-cl(1)),elc))                    1935
      return                                                               1936
      end                                                                  1937
      function nintot(ni,nx,nc,a,m,nin,is)                                 1938
      real a(nx,nc)                                                        1938
      integer m(nx),is(ni)                                                 1939
      is=0                                                                 1939
      nintot=0                                                             1940
14560 do 14561 ic=1,nc                                                     1940
14570 do 14571 j=1,nin                                                     1940
      k=m(j)                                                               1940
      if(is(k).ne.0)goto 14571                                             1941
      if(a(j,ic).eq.0.0)goto 14571                                         1941
      is(k)=k                                                              1941
      nintot=nintot+1                                                      1942
14571 continue                                                             1942
14572 continue                                                             1942
14561 continue                                                             1943
14562 continue                                                             1943
      return                                                               1944
      end                                                                  1945
      subroutine luncomp(ni,nx,nc,ca,ia,nin,a)                             1946
      real ca(nx,nc),a(ni,nc)                                              1946
      integer ia(nx)                                                       1947
      a=0.0                                                                1948
14580 do 14581 ic=1,nc                                                     1948
      if(nin.gt.0) a(ia(1:nin),ic)=ca(1:nin,ic)                            1948
14581 continue                                                             1949
14582 continue                                                             1949
      return                                                               1950
      end                                                                  1951
      subroutine lmodval(nt,x,nc,nx,a0,ca,ia,nin,ans)                      1952
      real a0(nc),ca(nx,nc),x(nt,*),ans(nc,nt)                             1952
      integer ia(nx)                                                       1953
14590 do 14591 i=1,nt                                                      1953
14600 do 14601 ic=1,nc                                                     1953
      ans(ic,i)=a0(ic)                                                     1955
      if(nin.gt.0) ans(ic,i)=ans(ic,i)+dot_product(ca(1:nin,ic),x(i,ia(1   1956 
     *:nin)))
14601 continue                                                             1956
14602 continue                                                             1956
14591 continue                                                             1957
14592 continue                                                             1957
      return                                                               1958
      end                                                                  1959
      subroutine splognet (parm,no,ni,nc,x,ix,jx,y,g,jd,vp,cl,ne,nx,nlam   1961 
     *,flmin,  ulam,thr,isd,intr,maxit,kopt,lmu,a0,ca,ia,nin,dev0,dev,al
     *m,nlp,jerr)
      real x(*),y(no,max(2,nc)),g(no,nc),vp(ni),ulam(nlam)                 1962
      real ca(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),cl(2,ni)         1963
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           1964
      real, dimension (:), allocatable :: xm,xs,ww,vq,xv                        
      integer, dimension (:), allocatable :: ju                                 
      if(maxval(vp) .gt. 0.0)goto 14621                                    1968
      jerr=10000                                                           1968
      return                                                               1968
14621 continue                                                             1969
      allocate(ww(1:no),stat=jerr)                                         1970
      allocate(ju(1:ni),stat=ierr)                                         1970
      jerr=jerr+ierr                                                       1971
      allocate(vq(1:ni),stat=ierr)                                         1971
      jerr=jerr+ierr                                                       1972
      allocate(xm(1:ni),stat=ierr)                                         1972
      jerr=jerr+ierr                                                       1973
      allocate(xs(1:ni),stat=ierr)                                         1973
      jerr=jerr+ierr                                                       1974
      if(kopt .ne. 2)goto 14641                                            1974
      allocate(xv(1:ni),stat=ierr)                                         1974
      jerr=jerr+ierr                                                       1974
14641 continue                                                             1975
      if(jerr.ne.0) return                                                 1976
      call spchkvars(no,ni,x,ix,ju)                                        1977
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 1978
      if(maxval(ju) .gt. 0)goto 14661                                      1978
      jerr=7777                                                            1978
      return                                                               1978
14661 continue                                                             1979
      vq=max(0.0,vp)                                                       1979
      vq=vq*ni/sum(vq)                                                     1980
14670 do 14671 i=1,no                                                      1980
      ww(i)=sum(y(i,:))                                                    1980
      if(ww(i).gt.0.0) y(i,:)=y(i,:)/ww(i)                                 1980
14671 continue                                                             1981
14672 continue                                                             1981
      sw=sum(ww)                                                           1981
      ww=ww/sw                                                             1982
      if(nc .ne. 1)goto 14691                                              1982
      call splstandard2(no,ni,x,ix,jx,ww,ju,isd,intr,xm,xs)                1983
      if(isd .le. 0)goto 14711                                             1983
14720 do 14721 j=1,ni                                                      1983
      cl(:,j)=cl(:,j)*xs(j)                                                1983
14721 continue                                                             1983
14722 continue                                                             1983
14711 continue                                                             1984
      call sprlognet2n(parm,no,ni,x,ix,jx,y(:,1),g(:,1),ww,ju,vq,cl,ne,n   1987 
     *x,nlam,  flmin,ulam,thr,isd,intr,maxit,kopt,xm,xs,lmu,a0,ca,ia,nin
     *,dev0,dev,  alm,nlp,jerr)
      goto 14681                                                           1988
14691 if(kopt .ne. 2)goto 14731                                            1989
      call multsplstandard2(no,ni,x,ix,jx,ww,ju,isd,intr,xm,xs,xv)         1990
      if(isd .le. 0)goto 14751                                             1990
14760 do 14761 j=1,ni                                                      1990
      cl(:,j)=cl(:,j)*xs(j)                                                1990
14761 continue                                                             1990
14762 continue                                                             1990
14751 continue                                                             1991
      call multsprlognetn(parm,no,ni,nc,x,ix,jx,y,g,ww,ju,vq,cl,ne,nx,nl   1993 
     *am,flmin,  ulam,thr,intr,maxit,xv,xm,xs,lmu,a0,ca,ia,nin,dev0,dev,
     *alm,nlp,jerr)
      goto 14771                                                           1994
14731 continue                                                             1994
      call splstandard2(no,ni,x,ix,jx,ww,ju,isd,intr,xm,xs)                1995
      if(isd .le. 0)goto 14791                                             1995
14800 do 14801 j=1,ni                                                      1995
      cl(:,j)=cl(:,j)*xs(j)                                                1995
14801 continue                                                             1995
14802 continue                                                             1995
14791 continue                                                             1996
      call sprlognetn(parm,no,ni,nc,x,ix,jx,y,g,ww,ju,vq,cl,ne,nx,nlam,f   1999 
     *lmin,  ulam,thr,isd,intr,maxit,kopt,xm,xs,lmu,a0,ca,  ia,nin,dev0,
     *dev,alm,nlp,jerr)
14771 continue                                                             2000
14681 continue                                                             2000
      if(jerr.gt.0) return                                                 2000
      dev0=2.0*sw*dev0                                                     2001
14810 do 14811 k=1,lmu                                                     2001
      nk=nin(k)                                                            2002
14820 do 14821 ic=1,nc                                                     2002
      if(isd .le. 0)goto 14841                                             2002
14850 do 14851 l=1,nk                                                      2002
      ca(l,ic,k)=ca(l,ic,k)/xs(ia(l))                                      2002
14851 continue                                                             2002
14852 continue                                                             2002
14841 continue                                                             2003
      if(intr .ne. 0)goto 14871                                            2003
      a0(ic,k)=0.0                                                         2003
      goto 14881                                                           2004
14871 continue                                                             2004
      a0(ic,k)=a0(ic,k)-dot_product(ca(1:nk,ic,k),xm(ia(1:nk)))            2004
14881 continue                                                             2005
14861 continue                                                             2005
14821 continue                                                             2006
14822 continue                                                             2006
14811 continue                                                             2007
14812 continue                                                             2007
      deallocate(ww,ju,vq,xm,xs)                                           2007
      if(kopt.eq.2) deallocate(xv)                                         2008
      return                                                               2009
      end                                                                  2010
      subroutine multsplstandard2(no,ni,x,ix,jx,w,ju,isd,intr,xm,xs,xv)    2011
      real x(*),w(no),xm(ni),xs(ni),xv(ni)                                 2011
      integer ix(*),jx(*),ju(ni)                                           2012
      if(intr .ne. 0)goto 14901                                            2013
14910 do 14911 j=1,ni                                                      2013
      if(ju(j).eq.0)goto 14911                                             2013
      xm(j)=0.0                                                            2013
      jb=ix(j)                                                             2013
      je=ix(j+1)-1                                                         2014
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)                          2015
      if(isd .eq. 0)goto 14931                                             2015
      xbq=dot_product(w(jx(jb:je)),x(jb:je))**2                            2015
      vc=xv(j)-xbq                                                         2016
      xs(j)=sqrt(vc)                                                       2016
      xv(j)=1.0+xbq/vc                                                     2017
      goto 14941                                                           2018
14931 continue                                                             2018
      xs(j)=1.0                                                            2018
14941 continue                                                             2019
14921 continue                                                             2019
14911 continue                                                             2020
14912 continue                                                             2020
      return                                                               2021
14901 continue                                                             2022
14950 do 14951 j=1,ni                                                      2022
      if(ju(j).eq.0)goto 14951                                             2022
      jb=ix(j)                                                             2022
      je=ix(j+1)-1                                                         2023
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             2024
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j)**2                 2025
      if(isd .le. 0)goto 14971                                             2025
      xs(j)=sqrt(xv(j))                                                    2025
      xv(j)=1.0                                                            2025
14971 continue                                                             2026
14951 continue                                                             2027
14952 continue                                                             2027
      if(isd.eq.0) xs=1.0                                                  2028
      return                                                               2029
      end                                                                  2030
      subroutine splstandard2(no,ni,x,ix,jx,w,ju,isd,intr,xm,xs)           2031
      real x(*),w(no),xm(ni),xs(ni)                                        2031
      integer ix(*),jx(*),ju(ni)                                           2032
      if(intr .ne. 0)goto 14991                                            2033
15000 do 15001 j=1,ni                                                      2033
      if(ju(j).eq.0)goto 15001                                             2033
      xm(j)=0.0                                                            2033
      jb=ix(j)                                                             2033
      je=ix(j+1)-1                                                         2034
      if(isd .eq. 0)goto 15021                                             2035
      vc=dot_product(w(jx(jb:je)),x(jb:je)**2)  -dot_product(w(jx(jb:je)   2037 
     *),x(jb:je))**2
      xs(j)=sqrt(vc)                                                       2038
      goto 15031                                                           2039
15021 continue                                                             2039
      xs(j)=1.0                                                            2039
15031 continue                                                             2040
15011 continue                                                             2040
15001 continue                                                             2041
15002 continue                                                             2041
      return                                                               2042
14991 continue                                                             2043
15040 do 15041 j=1,ni                                                      2043
      if(ju(j).eq.0)goto 15041                                             2043
      jb=ix(j)                                                             2043
      je=ix(j+1)-1                                                         2044
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             2045
      if(isd.ne.0) xs(j)=sqrt(dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j   2046 
     *)**2)
15041 continue                                                             2047
15042 continue                                                             2047
      if(isd.eq.0) xs=1.0                                                  2048
      return                                                               2049
      end                                                                  2050
      subroutine sprlognet2n (parm,no,ni,x,ix,jx,y,g,w,ju,vp,cl,ne,nx,nl   2053 
     *am,  flmin,ulam,shri,isd,intr,maxit,kopt,xb,xs,  lmu,a0,a,m,kin,de
     *v0,dev,alm,nlp,jerr)
      real x(*),y(no),g(no),w(no),vp(ni),ulam(nlam),cl(2,ni)               2054
      real a(nx,nlam),a0(nlam),dev(nlam),alm(nlam)                         2055
      real xb(ni),xs(ni)                                                   2055
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)                           2056
      real, dimension (:), allocatable :: xm,b,bs,v,r,sc,xv,q,ga                
      integer, dimension (:), allocatable :: mm,ixx                             
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               2061
      allocate(b(0:ni),stat=jerr)                                          2062
      allocate(xm(0:ni),stat=ierr)                                         2062
      jerr=jerr+ierr                                                       2063
      allocate(xv(1:ni),stat=ierr)                                         2063
      jerr=jerr+ierr                                                       2064
      allocate(bs(0:ni),stat=ierr)                                         2064
      jerr=jerr+ierr                                                       2065
      allocate(ga(1:ni),stat=ierr)                                         2065
      jerr=jerr+ierr                                                       2066
      allocate(mm(1:ni),stat=ierr)                                         2066
      jerr=jerr+ierr                                                       2067
      allocate(ixx(1:ni),stat=ierr)                                        2067
      jerr=jerr+ierr                                                       2068
      allocate(q(1:no),stat=ierr)                                          2068
      jerr=jerr+ierr                                                       2069
      allocate(r(1:no),stat=ierr)                                          2069
      jerr=jerr+ierr                                                       2070
      allocate(v(1:no),stat=ierr)                                          2070
      jerr=jerr+ierr                                                       2071
      allocate(sc(1:no),stat=ierr)                                         2071
      jerr=jerr+ierr                                                       2072
      if(jerr.ne.0) return                                                 2073
      fmax=log(1.0/pmin-1.0)                                               2073
      fmin=-fmax                                                           2073
      vmin=(1.0+pmin)*pmin*(1.0-pmin)                                      2074
      bta=parm                                                             2074
      omb=1.0-bta                                                          2075
      q0=dot_product(w,y)                                                  2075
      if(q0 .gt. pmin)goto 15061                                           2075
      jerr=8001                                                            2075
      return                                                               2075
15061 continue                                                             2076
      if(q0 .lt. 1.0-pmin)goto 15081                                       2076
      jerr=9001                                                            2076
      return                                                               2076
15081 continue                                                             2077
      if(intr.eq.0) q0=0.5                                                 2077
      bz=0.0                                                               2077
      if(intr.ne.0) bz=log(q0/(1.0-q0))                                    2078
      if(nonzero(no,g) .ne. 0)goto 15101                                   2078
      vi=q0*(1.0-q0)                                                       2078
      b(0)=bz                                                              2078
      v=vi*w                                                               2079
      r=w*(y-q0)                                                           2079
      q=q0                                                                 2079
      xm(0)=vi                                                             2079
      dev1=-(bz*q0+log(1.0-q0))                                            2080
      goto 15111                                                           2081
15101 continue                                                             2081
      b(0)=0.0                                                             2082
      if(intr .eq. 0)goto 15131                                            2082
      b(0)=azero(no,y,g,w,jerr)                                            2082
      if(jerr.ne.0) return                                                 2082
15131 continue                                                             2083
      q=1.0/(1.0+exp(-b(0)-g))                                             2083
      v=w*q*(1.0-q)                                                        2083
      r=w*(y-q)                                                            2083
      xm(0)=sum(v)                                                         2084
      dev1=-(b(0)*q0+dot_product(w,y*g+log(1.0-q)))                        2085
15111 continue                                                             2086
15091 continue                                                             2086
      if(kopt .le. 0)goto 15151                                            2087
      if(isd .le. 0 .or. intr .eq. 0)goto 15171                            2087
      xv=0.25                                                              2087
      goto 15181                                                           2088
15171 continue                                                             2089
15190 do 15191 j=1,ni                                                      2089
      if(ju(j).eq.0)goto 15191                                             2089
      jb=ix(j)                                                             2089
      je=ix(j+1)-1                                                         2090
      xv(j)=0.25*(dot_product(w(jx(jb:je)),x(jb:je)**2)-xb(j)**2)          2091
15191 continue                                                             2092
15192 continue                                                             2092
15181 continue                                                             2093
15161 continue                                                             2093
15151 continue                                                             2094
      b(1:ni)=0.0                                                          2094
      dev0=dev1                                                            2095
15200 do 15201 i=1,no                                                      2095
      if(y(i).gt.0.0) dev0=dev0+w(i)*y(i)*log(y(i))                        2096
      if(y(i).lt.1.0) dev0=dev0+w(i)*(1.0-y(i))*log(1.0-y(i))              2097
15201 continue                                                             2098
15202 continue                                                             2098
      if(flmin .ge. 1.0)goto 15221                                         2098
      eqs=max(eps,flmin)                                                   2098
      alf=eqs**(1.0/(nlam-1))                                              2098
15221 continue                                                             2099
      m=0                                                                  2099
      mm=0                                                                 2099
      nin=0                                                                2099
      o=0.0                                                                2099
      svr=o                                                                2099
      mnl=min(mnlam,nlam)                                                  2099
      bs=0.0                                                               2099
      nlp=0                                                                2099
      nin=nlp                                                              2100
      shr=shri*dev0                                                        2100
      al=0.0                                                               2100
      ixx=0                                                                2101
15230 do 15231 j=1,ni                                                      2101
      if(ju(j).eq.0)goto 15231                                             2102
      jb=ix(j)                                                             2102
      je=ix(j+1)-1                                                         2102
      jn=ix(j+1)-ix(j)                                                     2103
      sc(1:jn)=r(jx(jb:je))+v(jx(jb:je))*o                                 2104
      gj=dot_product(sc(1:jn),x(jb:je))                                    2105
      ga(j)=abs((gj-svr*xb(j))/xs(j))                                      2106
15231 continue                                                             2107
15232 continue                                                             2107
15240 do 15241 ilm=1,nlam                                                  2107
      al0=al                                                               2108
      if(flmin .lt. 1.0)goto 15261                                         2108
      al=ulam(ilm)                                                         2108
      goto 15251                                                           2109
15261 if(ilm .le. 2)goto 15271                                             2109
      al=al*alf                                                            2109
      goto 15251                                                           2110
15271 if(ilm .ne. 1)goto 15281                                             2110
      al=big                                                               2110
      goto 15291                                                           2111
15281 continue                                                             2111
      al0=0.0                                                              2112
15300 do 15301 j=1,ni                                                      2112
      if(ju(j).eq.0)goto 15301                                             2112
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            2112
15301 continue                                                             2113
15302 continue                                                             2113
      al0=al0/max(bta,1.0e-3)                                              2113
      al=alf*al0                                                           2114
15291 continue                                                             2115
15251 continue                                                             2115
      al2=al*omb                                                           2115
      al1=al*bta                                                           2115
      tlam=bta*(2.0*al-al0)                                                2116
15310 do 15311 k=1,ni                                                      2116
      if(ixx(k).eq.1)goto 15311                                            2116
      if(ju(k).eq.0)goto 15311                                             2117
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     2118
15311 continue                                                             2119
15312 continue                                                             2119
10880 continue                                                             2120
15320 continue                                                             2120
15321 continue                                                             2120
      bs(0)=b(0)                                                           2120
      if(nin.gt.0) bs(m(1:nin))=b(m(1:nin))                                2121
15330 do 15331 j=1,ni                                                      2121
      if(ixx(j).eq.0)goto 15331                                            2122
      jb=ix(j)                                                             2122
      je=ix(j+1)-1                                                         2122
      jn=ix(j+1)-ix(j)                                                     2123
      sc(1:jn)=v(jx(jb:je))                                                2124
      xm(j)=dot_product(sc(1:jn),x(jb:je))                                 2125
      if(kopt .ne. 0)goto 15351                                            2126
      xv(j)=dot_product(sc(1:jn),x(jb:je)**2)                              2127
      xv(j)=(xv(j)-2.0*xb(j)*xm(j)+xm(0)*xb(j)**2)/xs(j)**2                2128
15351 continue                                                             2129
15331 continue                                                             2130
15332 continue                                                             2130
15360 continue                                                             2130
15361 continue                                                             2130
      nlp=nlp+1                                                            2130
      dlx=0.0                                                              2131
15370 do 15371 k=1,ni                                                      2131
      if(ixx(k).eq.0)goto 15371                                            2132
      jb=ix(k)                                                             2132
      je=ix(k+1)-1                                                         2132
      jn=ix(k+1)-ix(k)                                                     2132
      bk=b(k)                                                              2133
      sc(1:jn)=r(jx(jb:je))+v(jx(jb:je))*o                                 2134
      gk=dot_product(sc(1:jn),x(jb:je))                                    2135
      gk=(gk-svr*xb(k))/xs(k)                                              2136
      u=gk+xv(k)*b(k)                                                      2136
      au=abs(u)-vp(k)*al1                                                  2137
      if(au .gt. 0.0)goto 15391                                            2137
      b(k)=0.0                                                             2137
      goto 15401                                                           2138
15391 continue                                                             2139
      b(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(xv(k)+vp(k)*al2)))          2140
15401 continue                                                             2141
15381 continue                                                             2141
      d=b(k)-bk                                                            2141
      if(abs(d).le.0.0)goto 15371                                          2141
      dlx=max(dlx,xv(k)*d**2)                                              2142
      if(mm(k) .ne. 0)goto 15421                                           2142
      nin=nin+1                                                            2142
      if(nin.gt.nx)goto 15372                                              2143
      mm(k)=nin                                                            2143
      m(nin)=k                                                             2143
      sc(1:jn)=v(jx(jb:je))                                                2144
      xm(k)=dot_product(sc(1:jn),x(jb:je))                                 2145
15421 continue                                                             2146
      r(jx(jb:je))=r(jx(jb:je))-d*v(jx(jb:je))*x(jb:je)/xs(k)              2147
      o=o+d*(xb(k)/xs(k))                                                  2148
      svr=svr-d*(xm(k)-xb(k)*xm(0))/xs(k)                                  2149
15371 continue                                                             2150
15372 continue                                                             2150
      if(nin.gt.nx)goto 15362                                              2151
      d=0.0                                                                2151
      if(intr.ne.0) d=svr/xm(0)                                            2152
      if(d .eq. 0.0)goto 15441                                             2152
      b(0)=b(0)+d                                                          2152
      dlx=max(dlx,xm(0)*d**2)                                              2152
      r=r-d*v                                                              2153
      svr=svr-d*xm(0)                                                      2154
15441 continue                                                             2155
      if(dlx.lt.shr)goto 15362                                             2156
      if(nlp .le. maxit)goto 15461                                         2156
      jerr=-ilm                                                            2156
      return                                                               2156
15461 continue                                                             2157
15470 continue                                                             2157
15471 continue                                                             2157
      nlp=nlp+1                                                            2157
      dlx=0.0                                                              2158
15480 do 15481 l=1,nin                                                     2158
      k=m(l)                                                               2158
      jb=ix(k)                                                             2158
      je=ix(k+1)-1                                                         2159
      jn=ix(k+1)-ix(k)                                                     2159
      bk=b(k)                                                              2160
      sc(1:jn)=r(jx(jb:je))+v(jx(jb:je))*o                                 2161
      gk=dot_product(sc(1:jn),x(jb:je))                                    2162
      gk=(gk-svr*xb(k))/xs(k)                                              2163
      u=gk+xv(k)*b(k)                                                      2163
      au=abs(u)-vp(k)*al1                                                  2164
      if(au .gt. 0.0)goto 15501                                            2164
      b(k)=0.0                                                             2164
      goto 15511                                                           2165
15501 continue                                                             2166
      b(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(xv(k)+vp(k)*al2)))          2167
15511 continue                                                             2168
15491 continue                                                             2168
      d=b(k)-bk                                                            2168
      if(abs(d).le.0.0)goto 15481                                          2168
      dlx=max(dlx,xv(k)*d**2)                                              2169
      r(jx(jb:je))=r(jx(jb:je))-d*v(jx(jb:je))*x(jb:je)/xs(k)              2170
      o=o+d*(xb(k)/xs(k))                                                  2171
      svr=svr-d*(xm(k)-xb(k)*xm(0))/xs(k)                                  2172
15481 continue                                                             2173
15482 continue                                                             2173
      d=0.0                                                                2173
      if(intr.ne.0) d=svr/xm(0)                                            2174
      if(d .eq. 0.0)goto 15531                                             2174
      b(0)=b(0)+d                                                          2174
      dlx=max(dlx,xm(0)*d**2)                                              2174
      r=r-d*v                                                              2175
      svr=svr-d*xm(0)                                                      2176
15531 continue                                                             2177
      if(dlx.lt.shr)goto 15472                                             2178
      if(nlp .le. maxit)goto 15551                                         2178
      jerr=-ilm                                                            2178
      return                                                               2178
15551 continue                                                             2179
      goto 15471                                                           2180
15472 continue                                                             2180
      goto 15361                                                           2181
15362 continue                                                             2181
      if(nin.gt.nx)goto 15322                                              2182
      sc=b(0)                                                              2182
      b0=0.0                                                               2183
15560 do 15561 j=1,nin                                                     2183
      l=m(j)                                                               2183
      jb=ix(l)                                                             2183
      je=ix(l+1)-1                                                         2184
      sc(jx(jb:je))=sc(jx(jb:je))+b(l)*x(jb:je)/xs(l)                      2185
      b0=b0-b(l)*xb(l)/xs(l)                                               2186
15561 continue                                                             2187
15562 continue                                                             2187
      sc=sc+b0                                                             2188
15570 do 15571 i=1,no                                                      2188
      fi=sc(i)+g(i)                                                        2189
      if(fi .ge. fmin)goto 15591                                           2189
      q(i)=0.0                                                             2189
      goto 15581                                                           2189
15591 if(fi .le. fmax)goto 15601                                           2189
      q(i)=1.0                                                             2189
      goto 15611                                                           2190
15601 continue                                                             2190
      q(i)=1.0/(1.0+exp(-fi))                                              2190
15611 continue                                                             2191
15581 continue                                                             2191
15571 continue                                                             2192
15572 continue                                                             2192
      v=w*q*(1.0-q)                                                        2192
      xm(0)=sum(v)                                                         2192
      if(xm(0).lt.vmin)goto 15322                                          2193
      r=w*(y-q)                                                            2193
      svr=sum(r)                                                           2193
      o=0.0                                                                2194
      if(xm(0)*(b(0)-bs(0))**2 .ge. shr)goto 15631                         2194
      kx=0                                                                 2195
15640 do 15641 j=1,nin                                                     2195
      k=m(j)                                                               2196
      if(xv(k)*(b(k)-bs(k))**2.lt.shr)goto 15641                           2196
      kx=1                                                                 2196
      goto 15642                                                           2197
15641 continue                                                             2198
15642 continue                                                             2198
      if(kx .ne. 0)goto 15661                                              2199
15670 do 15671 j=1,ni                                                      2199
      if(ixx(j).eq.1)goto 15671                                            2199
      if(ju(j).eq.0)goto 15671                                             2200
      jb=ix(j)                                                             2200
      je=ix(j+1)-1                                                         2200
      jn=ix(j+1)-ix(j)                                                     2201
      sc(1:jn)=r(jx(jb:je))+v(jx(jb:je))*o                                 2202
      gj=dot_product(sc(1:jn),x(jb:je))                                    2203
      ga(j)=abs((gj-svr*xb(j))/xs(j))                                      2204
      if(ga(j) .le. al1*vp(j))goto 15691                                   2204
      ixx(j)=1                                                             2204
      kx=1                                                                 2204
15691 continue                                                             2205
15671 continue                                                             2206
15672 continue                                                             2206
      if(kx.eq.1) go to 10880                                              2207
      goto 15322                                                           2208
15661 continue                                                             2209
15631 continue                                                             2210
      goto 15321                                                           2211
15322 continue                                                             2211
      if(nin .le. nx)goto 15711                                            2211
      jerr=-10000-ilm                                                      2211
      goto 15242                                                           2211
15711 continue                                                             2212
      if(nin.gt.0) a(1:nin,ilm)=b(m(1:nin))                                2212
      kin(ilm)=nin                                                         2213
      a0(ilm)=b(0)                                                         2213
      alm(ilm)=al                                                          2213
      lmu=ilm                                                              2214
      devi=dev2(no,w,y,q,pmin)                                             2215
      dev(ilm)=(dev1-devi)/dev0                                            2216
      if(ilm.lt.mnl)goto 15241                                             2216
      if(flmin.ge.1.0)goto 15241                                           2217
      me=0                                                                 2217
15720 do 15721 j=1,nin                                                     2217
      if(a(j,ilm).ne.0.0) me=me+1                                          2217
15721 continue                                                             2217
15722 continue                                                             2217
      if(me.gt.ne)goto 15242                                               2218
      if(dev(ilm).gt.devmax)goto 15242                                     2218
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 15242                             2219
      if(xm(0).lt.vmin)goto 15242                                          2220
15241 continue                                                             2221
15242 continue                                                             2221
      g=log(q/(1.0-q))                                                     2222
      deallocate(xm,b,bs,v,r,sc,xv,q,mm,ga,ixx)                            2223
      return                                                               2224
      end                                                                  2225
      subroutine sprlognetn(parm,no,ni,nc,x,ix,jx,y,g,w,ju,vp,cl,ne,nx,n   2227 
     *lam,flmin,  ulam,shri,isd,intr,maxit,kopt,xb,xs,lmu,a0,a,m,kin,dev
     *0,dev,alm,nlp,jerr)
      real x(*),y(no,nc),g(no,nc),w(no),vp(ni),ulam(nlam),xb(ni),xs(ni)    2228
      real a(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),cl(2,ni)          2229
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)                           2230
      real, dimension (:,:), allocatable :: q                                   
      real, dimension (:), allocatable :: sxp,sxpl                              
      real, dimension (:), allocatable :: sc,xm,v,r,ga                          
      real, dimension (:,:), allocatable :: b,bs,xv                             
      integer, dimension (:), allocatable :: mm,is,iy                           
      allocate(b(0:ni,1:nc),stat=jerr)                                          
      allocate(xv(1:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(bs(0:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(q(1:no,1:nc),stat=ierr); jerr=jerr+ierr                          
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               2241
      exmn=-exmx                                                           2242
      allocate(xm(0:ni),stat=ierr)                                         2242
      jerr=jerr+ierr                                                       2243
      allocate(r(1:no),stat=ierr)                                          2243
      jerr=jerr+ierr                                                       2244
      allocate(v(1:no),stat=ierr)                                          2244
      jerr=jerr+ierr                                                       2245
      allocate(mm(1:ni),stat=ierr)                                         2245
      jerr=jerr+ierr                                                       2246
      allocate(ga(1:ni),stat=ierr)                                         2246
      jerr=jerr+ierr                                                       2247
      allocate(iy(1:ni),stat=ierr)                                         2247
      jerr=jerr+ierr                                                       2248
      allocate(is(1:max(nc,ni)),stat=ierr)                                 2248
      jerr=jerr+ierr                                                       2249
      allocate(sxp(1:no),stat=ierr)                                        2249
      jerr=jerr+ierr                                                       2250
      allocate(sxpl(1:no),stat=ierr)                                       2250
      jerr=jerr+ierr                                                       2251
      allocate(sc(1:no),stat=ierr)                                         2251
      jerr=jerr+ierr                                                       2252
      if(jerr.ne.0) return                                                 2253
      pmax=1.0-pmin                                                        2253
      emin=pmin/pmax                                                       2253
      emax=1.0/emin                                                        2254
      pfm=(1.0+pmin)*pmin                                                  2254
      pfx=(1.0-pmin)*pmax                                                  2254
      vmin=pfm*pmax                                                        2255
      bta=parm                                                             2255
      omb=1.0-bta                                                          2255
      dev1=0.0                                                             2255
      dev0=0.0                                                             2256
15730 do 15731 ic=1,nc                                                     2256
      q0=dot_product(w,y(:,ic))                                            2257
      if(q0 .gt. pmin)goto 15751                                           2257
      jerr =8000+ic                                                        2257
      return                                                               2257
15751 continue                                                             2258
      if(q0 .lt. 1.0-pmin)goto 15771                                       2258
      jerr =9000+ic                                                        2258
      return                                                               2258
15771 continue                                                             2259
      if(intr.eq.0) q0=1.0/nc                                              2260
      b(1:ni,ic)=0.0                                                       2260
      b(0,ic)=0.0                                                          2261
      if(intr .eq. 0)goto 15791                                            2261
      b(0,ic)=log(q0)                                                      2261
      dev1=dev1-q0*b(0,ic)                                                 2261
15791 continue                                                             2262
15731 continue                                                             2263
15732 continue                                                             2263
      if(intr.eq.0) dev1=log(float(nc))                                    2263
      iy=0                                                                 2263
      al=0.0                                                               2264
      if(nonzero(no*nc,g) .ne. 0)goto 15811                                2265
      b(0,:)=b(0,:)-sum(b(0,:))/nc                                         2265
      sxp=0.0                                                              2266
15820 do 15821 ic=1,nc                                                     2266
      q(:,ic)=exp(b(0,ic))                                                 2266
      sxp=sxp+q(:,ic)                                                      2266
15821 continue                                                             2267
15822 continue                                                             2267
      goto 15831                                                           2268
15811 continue                                                             2268
15840 do 15841 i=1,no                                                      2268
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         2268
15841 continue                                                             2268
15842 continue                                                             2268
      sxp=0.0                                                              2269
      if(intr .ne. 0)goto 15861                                            2269
      b(0,:)=0.0                                                           2269
      goto 15871                                                           2270
15861 continue                                                             2270
      call kazero(nc,no,y,g,w,b(0,:),jerr)                                 2270
      if(jerr.ne.0) return                                                 2270
15871 continue                                                             2271
15851 continue                                                             2271
      dev1=0.0                                                             2272
15880 do 15881 ic=1,nc                                                     2272
      q(:,ic)=b(0,ic)+g(:,ic)                                              2273
      dev1=dev1-dot_product(w,y(:,ic)*q(:,ic))                             2274
      q(:,ic)=exp(q(:,ic))                                                 2274
      sxp=sxp+q(:,ic)                                                      2275
15881 continue                                                             2276
15882 continue                                                             2276
      sxpl=w*log(sxp)                                                      2276
15890 do 15891 ic=1,nc                                                     2276
      dev1=dev1+dot_product(y(:,ic),sxpl)                                  2276
15891 continue                                                             2277
15892 continue                                                             2277
15831 continue                                                             2278
15801 continue                                                             2278
15900 do 15901 ic=1,nc                                                     2278
15910 do 15911 i=1,no                                                      2278
      if(y(i,ic).gt.0.0) dev0=dev0+w(i)*y(i,ic)*log(y(i,ic))               2278
15911 continue                                                             2278
15912 continue                                                             2278
15901 continue                                                             2279
15902 continue                                                             2279
      dev0=dev0+dev1                                                       2280
      if(kopt .le. 0)goto 15931                                            2281
      if(isd .le. 0 .or. intr .eq. 0)goto 15951                            2281
      xv=0.25                                                              2281
      goto 15961                                                           2282
15951 continue                                                             2283
15970 do 15971 j=1,ni                                                      2283
      if(ju(j).eq.0)goto 15971                                             2283
      jb=ix(j)                                                             2283
      je=ix(j+1)-1                                                         2284
      xv(j,:)=0.25*(dot_product(w(jx(jb:je)),x(jb:je)**2)-xb(j)**2)        2285
15971 continue                                                             2286
15972 continue                                                             2286
15961 continue                                                             2287
15941 continue                                                             2287
15931 continue                                                             2288
      if(flmin .ge. 1.0)goto 15991                                         2288
      eqs=max(eps,flmin)                                                   2288
      alf=eqs**(1.0/(nlam-1))                                              2288
15991 continue                                                             2289
      m=0                                                                  2289
      mm=0                                                                 2289
      nin=0                                                                2289
      nlp=0                                                                2289
      mnl=min(mnlam,nlam)                                                  2289
      bs=0.0                                                               2289
      svr=0.0                                                              2289
      o=0.0                                                                2290
      shr=shri*dev0                                                        2290
      ga=0.0                                                               2291
16000 do 16001 ic=1,nc                                                     2291
      v=q(:,ic)/sxp                                                        2291
      r=w*(y(:,ic)-v)                                                      2291
      v=w*v*(1.0-v)                                                        2292
16010 do 16011 j=1,ni                                                      2292
      if(ju(j).eq.0)goto 16011                                             2293
      jb=ix(j)                                                             2293
      je=ix(j+1)-1                                                         2293
      jn=ix(j+1)-ix(j)                                                     2294
      sc(1:jn)=r(jx(jb:je))+o*v(jx(jb:je))                                 2295
      gj=dot_product(sc(1:jn),x(jb:je))                                    2296
      ga(j)=max(ga(j),abs(gj-svr*xb(j))/xs(j))                             2297
16011 continue                                                             2298
16012 continue                                                             2298
16001 continue                                                             2299
16002 continue                                                             2299
16020 do 16021 ilm=1,nlam                                                  2299
      al0=al                                                               2300
      if(flmin .lt. 1.0)goto 16041                                         2300
      al=ulam(ilm)                                                         2300
      goto 16031                                                           2301
16041 if(ilm .le. 2)goto 16051                                             2301
      al=al*alf                                                            2301
      goto 16031                                                           2302
16051 if(ilm .ne. 1)goto 16061                                             2302
      al=big                                                               2302
      goto 16071                                                           2303
16061 continue                                                             2303
      al0=0.0                                                              2304
16080 do 16081 j=1,ni                                                      2304
      if(ju(j).eq.0)goto 16081                                             2304
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            2304
16081 continue                                                             2305
16082 continue                                                             2305
      al0=al0/max(bta,1.0e-3)                                              2305
      al=alf*al0                                                           2306
16071 continue                                                             2307
16031 continue                                                             2307
      al2=al*omb                                                           2307
      al1=al*bta                                                           2307
      tlam=bta*(2.0*al-al0)                                                2308
16090 do 16091 k=1,ni                                                      2308
      if(iy(k).eq.1)goto 16091                                             2308
      if(ju(k).eq.0)goto 16091                                             2309
      if(ga(k).gt.tlam*vp(k)) iy(k)=1                                      2310
16091 continue                                                             2311
16092 continue                                                             2311
10880 continue                                                             2312
16100 continue                                                             2312
16101 continue                                                             2312
      ixx=0                                                                2312
      jxx=ixx                                                              2312
      ig=0                                                                 2313
16110 do 16111 ic=1,nc                                                     2313
      bs(0,ic)=b(0,ic)                                                     2314
      if(nin.gt.0) bs(m(1:nin),ic)=b(m(1:nin),ic)                          2315
      xm(0)=0.0                                                            2315
      svr=0.0                                                              2315
      o=0.0                                                                2316
16120 do 16121 i=1,no                                                      2316
      pic=q(i,ic)/sxp(i)                                                   2317
      if(pic .ge. pfm)goto 16141                                           2317
      pic=0.0                                                              2317
      v(i)=0.0                                                             2317
      goto 16131                                                           2318
16141 if(pic .le. pfx)goto 16151                                           2318
      pic=1.0                                                              2318
      v(i)=0.0                                                             2318
      goto 16161                                                           2319
16151 continue                                                             2319
      v(i)=w(i)*pic*(1.0-pic)                                              2319
      xm(0)=xm(0)+v(i)                                                     2319
16161 continue                                                             2320
16131 continue                                                             2320
      r(i)=w(i)*(y(i,ic)-pic)                                              2320
      svr=svr+r(i)                                                         2321
16121 continue                                                             2322
16122 continue                                                             2322
      if(xm(0).le.vmin)goto 16111                                          2322
      ig=1                                                                 2323
16170 do 16171 j=1,ni                                                      2323
      if(iy(j).eq.0)goto 16171                                             2324
      jb=ix(j)                                                             2324
      je=ix(j+1)-1                                                         2325
      xm(j)=dot_product(v(jx(jb:je)),x(jb:je))                             2326
      if(kopt .ne. 0)goto 16191                                            2327
      xv(j,ic)=dot_product(v(jx(jb:je)),x(jb:je)**2)                       2328
      xv(j,ic)=(xv(j,ic)-2.0*xb(j)*xm(j)+xm(0)*xb(j)**2)/xs(j)**2          2329
16191 continue                                                             2330
16171 continue                                                             2331
16172 continue                                                             2331
16200 continue                                                             2331
16201 continue                                                             2331
      nlp=nlp+1                                                            2331
      dlx=0.0                                                              2332
16210 do 16211 k=1,ni                                                      2332
      if(iy(k).eq.0)goto 16211                                             2333
      jb=ix(k)                                                             2333
      je=ix(k+1)-1                                                         2333
      jn=ix(k+1)-ix(k)                                                     2333
      bk=b(k,ic)                                                           2334
      sc(1:jn)=r(jx(jb:je))+o*v(jx(jb:je))                                 2335
      gk=dot_product(sc(1:jn),x(jb:je))                                    2336
      gk=(gk-svr*xb(k))/xs(k)                                              2337
      u=gk+xv(k,ic)*b(k,ic)                                                2337
      au=abs(u)-vp(k)*al1                                                  2338
      if(au .gt. 0.0)goto 16231                                            2338
      b(k,ic)=0.0                                                          2338
      goto 16241                                                           2339
16231 continue                                                             2340
      b(k,ic)=max(cl(1,k),min(cl(2,k),sign(au,u)/  (xv(k,ic)+vp(k)*al2))   2342 
     *)
16241 continue                                                             2343
16221 continue                                                             2343
      d=b(k,ic)-bk                                                         2343
      if(abs(d).le.0.0)goto 16211                                          2344
      dlx=max(dlx,xv(k,ic)*d**2)                                           2345
      if(mm(k) .ne. 0)goto 16261                                           2345
      nin=nin+1                                                            2346
      if(nin .le. nx)goto 16281                                            2346
      jxx=1                                                                2346
      goto 16212                                                           2346
16281 continue                                                             2347
      mm(k)=nin                                                            2347
      m(nin)=k                                                             2348
      xm(k)=dot_product(v(jx(jb:je)),x(jb:je))                             2349
16261 continue                                                             2350
      r(jx(jb:je))=r(jx(jb:je))-d*v(jx(jb:je))*x(jb:je)/xs(k)              2351
      o=o+d*(xb(k)/xs(k))                                                  2352
      svr=svr-d*(xm(k)-xb(k)*xm(0))/xs(k)                                  2353
16211 continue                                                             2354
16212 continue                                                             2354
      if(jxx.gt.0)goto 16202                                               2355
      d=0.0                                                                2355
      if(intr.ne.0) d=svr/xm(0)                                            2356
      if(d .eq. 0.0)goto 16301                                             2356
      b(0,ic)=b(0,ic)+d                                                    2356
      dlx=max(dlx,xm(0)*d**2)                                              2357
      r=r-d*v                                                              2357
      svr=svr-d*xm(0)                                                      2358
16301 continue                                                             2359
      if(dlx.lt.shr)goto 16202                                             2359
      if(nlp .le. maxit)goto 16321                                         2359
      jerr=-ilm                                                            2359
      return                                                               2359
16321 continue                                                             2360
16330 continue                                                             2360
16331 continue                                                             2360
      nlp=nlp+1                                                            2360
      dlx=0.0                                                              2361
16340 do 16341 l=1,nin                                                     2361
      k=m(l)                                                               2361
      jb=ix(k)                                                             2361
      je=ix(k+1)-1                                                         2362
      jn=ix(k+1)-ix(k)                                                     2362
      bk=b(k,ic)                                                           2363
      sc(1:jn)=r(jx(jb:je))+o*v(jx(jb:je))                                 2364
      gk=dot_product(sc(1:jn),x(jb:je))                                    2365
      gk=(gk-svr*xb(k))/xs(k)                                              2366
      u=gk+xv(k,ic)*b(k,ic)                                                2366
      au=abs(u)-vp(k)*al1                                                  2367
      if(au .gt. 0.0)goto 16361                                            2367
      b(k,ic)=0.0                                                          2367
      goto 16371                                                           2368
16361 continue                                                             2369
      b(k,ic)=max(cl(1,k),min(cl(2,k),sign(au,u)/  (xv(k,ic)+vp(k)*al2))   2371 
     *)
16371 continue                                                             2372
16351 continue                                                             2372
      d=b(k,ic)-bk                                                         2372
      if(abs(d).le.0.0)goto 16341                                          2373
      dlx=max(dlx,xv(k,ic)*d**2)                                           2374
      r(jx(jb:je))=r(jx(jb:je))-d*v(jx(jb:je))*x(jb:je)/xs(k)              2375
      o=o+d*(xb(k)/xs(k))                                                  2376
      svr=svr-d*(xm(k)-xb(k)*xm(0))/xs(k)                                  2377
16341 continue                                                             2378
16342 continue                                                             2378
      d=0.0                                                                2378
      if(intr.ne.0) d=svr/xm(0)                                            2379
      if(d .eq. 0.0)goto 16391                                             2379
      b(0,ic)=b(0,ic)+d                                                    2379
      dlx=max(dlx,xm(0)*d**2)                                              2380
      r=r-d*v                                                              2380
      svr=svr-d*xm(0)                                                      2381
16391 continue                                                             2382
      if(dlx.lt.shr)goto 16332                                             2382
      if(nlp .le. maxit)goto 16411                                         2382
      jerr=-ilm                                                            2382
      return                                                               2382
16411 continue                                                             2383
      goto 16331                                                           2384
16332 continue                                                             2384
      goto 16201                                                           2385
16202 continue                                                             2385
      if(jxx.gt.0)goto 16112                                               2386
      if(xm(0)*(b(0,ic)-bs(0,ic))**2.gt.shr) ixx=1                         2387
      if(ixx .ne. 0)goto 16431                                             2388
16440 do 16441 j=1,nin                                                     2388
      k=m(j)                                                               2389
      if(xv(k,ic)*(b(k,ic)-bs(k,ic))**2 .le. shr)goto 16461                2389
      ixx=1                                                                2389
      goto 16442                                                           2389
16461 continue                                                             2390
16441 continue                                                             2391
16442 continue                                                             2391
16431 continue                                                             2392
      sc=b(0,ic)+g(:,ic)                                                   2392
      b0=0.0                                                               2393
16470 do 16471 j=1,nin                                                     2393
      l=m(j)                                                               2393
      jb=ix(l)                                                             2393
      je=ix(l+1)-1                                                         2394
      sc(jx(jb:je))=sc(jx(jb:je))+b(l,ic)*x(jb:je)/xs(l)                   2395
      b0=b0-b(l,ic)*xb(l)/xs(l)                                            2396
16471 continue                                                             2397
16472 continue                                                             2397
      sc=min(max(exmn,sc+b0),exmx)                                         2398
      sxp=sxp-q(:,ic)                                                      2399
      q(:,ic)=min(max(emin*sxp,exp(sc)),emax*sxp)                          2400
      sxp=sxp+q(:,ic)                                                      2401
16111 continue                                                             2402
16112 continue                                                             2402
      s=-sum(b(0,:))/nc                                                    2402
      b(0,:)=b(0,:)+s                                                      2402
      sc=s                                                                 2402
      b0=0.0                                                               2403
16480 do 16481 j=1,nin                                                     2403
      l=m(j)                                                               2404
      if(vp(l) .gt. 0.0)goto 16501                                         2404
      s=sum(b(l,:))/nc                                                     2404
      goto 16511                                                           2405
16501 continue                                                             2405
      s=elc(parm,nc,cl(:,l),b(l,:),is)                                     2405
16511 continue                                                             2406
16491 continue                                                             2406
      b(l,:)=b(l,:)-s                                                      2407
      jb=ix(l)                                                             2407
      je=ix(l+1)-1                                                         2408
      sc(jx(jb:je))=sc(jx(jb:je))-s*x(jb:je)/xs(l)                         2409
      b0=b0+s*xb(l)/xs(l)                                                  2410
16481 continue                                                             2411
16482 continue                                                             2411
      sc=sc+b0                                                             2411
      sc=exp(sc)                                                           2411
      sxp=sxp*sc                                                           2411
16520 do 16521 ic=1,nc                                                     2411
      q(:,ic)=q(:,ic)*sc                                                   2411
16521 continue                                                             2412
16522 continue                                                             2412
      if(jxx.gt.0)goto 16102                                               2412
      if(ig.eq.0)goto 16102                                                2413
      if(ixx .ne. 0)goto 16541                                             2414
16550 do 16551 j=1,ni                                                      2414
      if(iy(j).eq.1)goto 16551                                             2414
      if(ju(j).eq.0)goto 16551                                             2414
      ga(j)=0.0                                                            2414
16551 continue                                                             2415
16552 continue                                                             2415
16560 do 16561 ic=1,nc                                                     2415
      v=q(:,ic)/sxp                                                        2415
      r=w*(y(:,ic)-v)                                                      2415
      v=w*v*(1.0-v)                                                        2416
16570 do 16571 j=1,ni                                                      2416
      if(iy(j).eq.1)goto 16571                                             2416
      if(ju(j).eq.0)goto 16571                                             2417
      jb=ix(j)                                                             2417
      je=ix(j+1)-1                                                         2417
      jn=ix(j+1)-ix(j)                                                     2418
      sc(1:jn)=r(jx(jb:je))+o*v(jx(jb:je))                                 2419
      gj=dot_product(sc(1:jn),x(jb:je))                                    2420
      ga(j)=max(ga(j),abs(gj-svr*xb(j))/xs(j))                             2421
16571 continue                                                             2422
16572 continue                                                             2422
16561 continue                                                             2423
16562 continue                                                             2423
16580 do 16581 k=1,ni                                                      2423
      if(iy(k).eq.1)goto 16581                                             2423
      if(ju(k).eq.0)goto 16581                                             2424
      if(ga(k) .le. al1*vp(k))goto 16601                                   2424
      iy(k)=1                                                              2424
      ixx=1                                                                2424
16601 continue                                                             2425
16581 continue                                                             2426
16582 continue                                                             2426
      if(ixx.eq.1) go to 10880                                             2427
      goto 16102                                                           2428
16541 continue                                                             2429
      goto 16101                                                           2430
16102 continue                                                             2430
      if(jxx .le. 0)goto 16621                                             2430
      jerr=-10000-ilm                                                      2430
      goto 16022                                                           2430
16621 continue                                                             2430
      devi=0.0                                                             2431
16630 do 16631 ic=1,nc                                                     2432
      if(nin.gt.0) a(1:nin,ic,ilm)=b(m(1:nin),ic)                          2432
      a0(ic,ilm)=b(0,ic)                                                   2433
16640 do 16641 i=1,no                                                      2433
      if(y(i,ic).le.0.0)goto 16641                                         2434
      devi=devi-w(i)*y(i,ic)*log(q(i,ic)/sxp(i))                           2435
16641 continue                                                             2436
16642 continue                                                             2436
16631 continue                                                             2437
16632 continue                                                             2437
      kin(ilm)=nin                                                         2437
      alm(ilm)=al                                                          2437
      lmu=ilm                                                              2438
      dev(ilm)=(dev1-devi)/dev0                                            2438
      if(ig.eq.0)goto 16022                                                2439
      if(ilm.lt.mnl)goto 16021                                             2439
      if(flmin.ge.1.0)goto 16021                                           2440
      if(nintot(ni,nx,nc,a(1,1,ilm),m,nin,is).gt.ne)goto 16022             2441
      if(dev(ilm).gt.devmax)goto 16022                                     2441
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 16022                             2442
16021 continue                                                             2443
16022 continue                                                             2443
      g=log(q)                                                             2443
16650 do 16651 i=1,no                                                      2443
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         2443
16651 continue                                                             2444
16652 continue                                                             2444
      deallocate(sxp,b,bs,v,r,xv,q,mm,is,xm,sc,ga,iy)                      2445
      return                                                               2446
      end                                                                  2447
      subroutine lcmodval(nc,nx,a0,ca,ia,nin,x,ix,jx,n,f)                  2448
      real a0(nc),ca(nx,nc),x(*),f(nc,n)                                   2448
      integer ia(*),ix(*),jx(*)                                            2449
16660 do 16661 ic=1,nc                                                     2449
      f(ic,:)=a0(ic)                                                       2449
16661 continue                                                             2450
16662 continue                                                             2450
16670 do 16671 j=1,nin                                                     2450
      k=ia(j)                                                              2450
      kb=ix(k)                                                             2450
      ke=ix(k+1)-1                                                         2451
16680 do 16681 ic=1,nc                                                     2451
      f(ic,jx(kb:ke))=f(ic,jx(kb:ke))+ca(j,ic)*x(kb:ke)                    2451
16681 continue                                                             2452
16682 continue                                                             2452
16671 continue                                                             2453
16672 continue                                                             2453
      return                                                               2454
      end                                                                  2455
      subroutine coxnet (parm,no,ni,x,y,d,g,w,jd,vp,cl,ne,nx,nlam,flmin,   2457 
     *ulam,thr,  maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      real x(no,ni),y(no),d(no),g(no),w(no),vp(ni),ulam(nlam)              2458
      real ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni)                        2459
      integer jd(*),ia(nx),nin(nlam)                                       2460
      real, dimension (:), allocatable :: xs,ww,vq                              
      integer, dimension (:), allocatable :: ju                                 
      if(maxval(vp) .gt. 0.0)goto 16701                                    2464
      jerr=10000                                                           2464
      return                                                               2464
16701 continue                                                             2465
      allocate(ww(1:no),stat=jerr)                                         2466
      allocate(ju(1:ni),stat=ierr)                                         2466
      jerr=jerr+ierr                                                       2467
      allocate(vq(1:ni),stat=ierr)                                         2467
      jerr=jerr+ierr                                                       2468
      if(isd .le. 0)goto 16721                                             2468
      allocate(xs(1:ni),stat=ierr)                                         2468
      jerr=jerr+ierr                                                       2468
16721 continue                                                             2469
      if(jerr.ne.0) return                                                 2470
      call chkvars(no,ni,x,ju)                                             2471
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 2472
      if(maxval(ju) .gt. 0)goto 16741                                      2472
      jerr=7777                                                            2472
      return                                                               2472
16741 continue                                                             2473
      vq=max(0.0,vp)                                                       2473
      vq=vq*ni/sum(vq)                                                     2474
      ww=max(0.0,w)                                                        2474
      sw=sum(ww)                                                           2475
      if(sw .gt. 0.0)goto 16761                                            2475
      jerr=9999                                                            2475
      return                                                               2475
16761 continue                                                             2475
      ww=ww/sw                                                             2476
      call cstandard(no,ni,x,ww,ju,isd,xs)                                 2477
      if(isd .le. 0)goto 16781                                             2477
16790 do 16791 j=1,ni                                                      2477
      cl(:,j)=cl(:,j)*xs(j)                                                2477
16791 continue                                                             2477
16792 continue                                                             2477
16781 continue                                                             2478
      call coxnet1(parm,no,ni,x,y,d,g,ww,ju,vq,cl,ne,nx,nlam,flmin,ulam,   2480 
     *thr,  isd,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return                                                 2480
      dev0=2.0*sw*dev0                                                     2481
      if(isd .le. 0)goto 16811                                             2481
16820 do 16821 k=1,lmu                                                     2481
      nk=nin(k)                                                            2481
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))                                   2481
16821 continue                                                             2481
16822 continue                                                             2481
16811 continue                                                             2482
      deallocate(ww,ju,vq)                                                 2482
      if(isd.gt.0) deallocate(xs)                                          2483
      return                                                               2484
      end                                                                  2485
      subroutine cstandard (no,ni,x,w,ju,isd,xs)                           2486
      real x(no,ni),w(no),xs(ni)                                           2486
      integer ju(ni)                                                       2487
16830 do 16831 j=1,ni                                                      2487
      if(ju(j).eq.0)goto 16831                                             2488
      xm=dot_product(w,x(:,j))                                             2488
      x(:,j)=x(:,j)-xm                                                     2489
      if(isd .le. 0)goto 16851                                             2489
      xs(j)=sqrt(dot_product(w,x(:,j)**2))                                 2489
      x(:,j)=x(:,j)/xs(j)                                                  2489
16851 continue                                                             2490
16831 continue                                                             2491
16832 continue                                                             2491
      return                                                               2492
      end                                                                  2493
      subroutine coxnet1(parm,no,ni,x,y,d,g,q,ju,vp,cl,ne,nx,nlam,flmin,   2495 
     *ulam,cthri,  isd,maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      real x(no,ni),y(no),q(no),d(no),g(no),vp(ni),ulam(nlam)              2496
      real ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni)                        2497
      integer ju(ni),m(nx),kin(nlam)                                       2498
      real, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq                
      real, dimension (:), allocatable :: e,uu,ga                               
      integer, dimension (:), allocatable :: jp,kp,mm,ixx                       
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               2504
      sml=sml*100.0                                                        2504
      devmax=devmax*0.99/0.999                                             2505
      allocate(e(1:no),stat=jerr)                                          2506
      allocate(uu(1:no),stat=ierr)                                         2506
      jerr=jerr+ierr                                                       2507
      allocate(f(1:no),stat=ierr)                                          2507
      jerr=jerr+ierr                                                       2508
      allocate(w(1:no),stat=ierr)                                          2508
      jerr=jerr+ierr                                                       2509
      allocate(v(1:ni),stat=ierr)                                          2509
      jerr=jerr+ierr                                                       2510
      allocate(a(1:ni),stat=ierr)                                          2510
      jerr=jerr+ierr                                                       2511
      allocate(as(1:ni),stat=ierr)                                         2511
      jerr=jerr+ierr                                                       2512
      allocate(xs(1:ni),stat=ierr)                                         2512
      jerr=jerr+ierr                                                       2513
      allocate(ga(1:ni),stat=ierr)                                         2513
      jerr=jerr+ierr                                                       2514
      allocate(ixx(1:ni),stat=ierr)                                        2514
      jerr=jerr+ierr                                                       2515
      allocate(jp(1:no),stat=ierr)                                         2515
      jerr=jerr+ierr                                                       2516
      allocate(kp(1:no),stat=ierr)                                         2516
      jerr=jerr+ierr                                                       2517
      allocate(dk(1:no),stat=ierr)                                         2517
      jerr=jerr+ierr                                                       2518
      allocate(wr(1:no),stat=ierr)                                         2518
      jerr=jerr+ierr                                                       2519
      allocate(dq(1:no),stat=ierr)                                         2519
      jerr=jerr+ierr                                                       2520
      allocate(mm(1:ni),stat=ierr)                                         2520
      jerr=jerr+ierr                                                       2521
      if(jerr.ne.0)go to 12180                                             2522
      call groups(no,y,d,q,nk,kp,jp,t0,jerr)                               2523
      if(jerr.ne.0) go to 12180                                            2523
      alpha=parm                                                           2524
      oma=1.0-alpha                                                        2524
      nlm=0                                                                2524
      ixx=0                                                                2524
      al=0.0                                                               2525
      dq=d*q                                                               2525
      call died(no,nk,dq,kp,jp,dk)                                         2526
      a=0.0                                                                2526
      f(1)=0.0                                                             2526
      fmax=log(huge(f(1))*0.1)                                             2527
      if(nonzero(no,g) .eq. 0)goto 16871                                   2527
      f=g-dot_product(q,g)                                                 2528
      e=q*exp(sign(min(abs(f),fmax),f))                                    2529
      goto 16881                                                           2530
16871 continue                                                             2530
      f=0.0                                                                2530
      e=q                                                                  2530
16881 continue                                                             2531
16861 continue                                                             2531
      r0=risk(no,ni,nk,dq,dk,f,e,kp,jp,uu)                                 2532
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)                         2532
      dev0=rr                                                              2533
16890 do 16891 i=1,no                                                      2533
      if((y(i) .ge. t0) .and. (q(i) .gt. 0.0))goto 16911                   2533
      w(i)=0.0                                                             2533
      wr(i)=w(i)                                                           2533
16911 continue                                                             2533
16891 continue                                                             2534
16892 continue                                                             2534
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)                         2535
      if(jerr.ne.0) go to 12180                                            2536
      if(flmin .ge. 1.0)goto 16931                                         2536
      eqs=max(eps,flmin)                                                   2536
      alf=eqs**(1.0/(nlam-1))                                              2536
16931 continue                                                             2537
      m=0                                                                  2537
      mm=0                                                                 2537
      nlp=0                                                                2537
      nin=nlp                                                              2537
      mnl=min(mnlam,nlam)                                                  2537
      as=0.0                                                               2537
      cthr=cthri*dev0                                                      2538
16940 do 16941 j=1,ni                                                      2538
      if(ju(j).eq.0)goto 16941                                             2538
      ga(j)=abs(dot_product(wr,x(:,j)))                                    2538
16941 continue                                                             2539
16942 continue                                                             2539
16950 do 16951 ilm=1,nlam                                                  2539
      al0=al                                                               2540
      if(flmin .lt. 1.0)goto 16971                                         2540
      al=ulam(ilm)                                                         2540
      goto 16961                                                           2541
16971 if(ilm .le. 2)goto 16981                                             2541
      al=al*alf                                                            2541
      goto 16961                                                           2542
16981 if(ilm .ne. 1)goto 16991                                             2542
      al=big                                                               2542
      goto 17001                                                           2543
16991 continue                                                             2543
      al0=0.0                                                              2544
17010 do 17011 j=1,ni                                                      2544
      if(ju(j).eq.0)goto 17011                                             2544
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            2544
17011 continue                                                             2545
17012 continue                                                             2545
      al0=al0/max(parm,1.0e-3)                                             2545
      al=alf*al0                                                           2546
17001 continue                                                             2547
16961 continue                                                             2547
      sa=alpha*al                                                          2547
      omal=oma*al                                                          2547
      tlam=alpha*(2.0*al-al0)                                              2548
17020 do 17021 k=1,ni                                                      2548
      if(ixx(k).eq.1)goto 17021                                            2548
      if(ju(k).eq.0)goto 17021                                             2549
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     2550
17021 continue                                                             2551
17022 continue                                                             2551
10880 continue                                                             2552
17030 continue                                                             2552
17031 continue                                                             2552
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))                                2553
      call vars(no,ni,x,w,ixx,v)                                           2554
17040 continue                                                             2554
17041 continue                                                             2554
      nlp=nlp+1                                                            2554
      dli=0.0                                                              2555
17050 do 17051 j=1,ni                                                      2555
      if(ixx(j).eq.0)goto 17051                                            2556
      u=a(j)*v(j)+dot_product(wr,x(:,j))                                   2557
      if(abs(u) .gt. vp(j)*sa)goto 17071                                   2557
      at=0.0                                                               2557
      goto 17081                                                           2558
17071 continue                                                             2558
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/  (v(j)+vp(j)*o   2560 
     *mal)))
17081 continue                                                             2561
17061 continue                                                             2561
      if(at .eq. a(j))goto 17101                                           2561
      del=at-a(j)                                                          2561
      a(j)=at                                                              2561
      dli=max(dli,v(j)*del**2)                                             2562
      wr=wr-del*w*x(:,j)                                                   2562
      f=f+del*x(:,j)                                                       2563
      if(mm(j) .ne. 0)goto 17121                                           2563
      nin=nin+1                                                            2563
      if(nin.gt.nx)goto 17052                                              2564
      mm(j)=nin                                                            2564
      m(nin)=j                                                             2565
17121 continue                                                             2566
17101 continue                                                             2567
17051 continue                                                             2568
17052 continue                                                             2568
      if(nin.gt.nx)goto 17042                                              2568
      if(dli.lt.cthr)goto 17042                                            2569
      if(nlp .le. maxit)goto 17141                                         2569
      jerr=-ilm                                                            2569
      return                                                               2569
17141 continue                                                             2570
17150 continue                                                             2570
17151 continue                                                             2570
      nlp=nlp+1                                                            2570
      dli=0.0                                                              2571
17160 do 17161 l=1,nin                                                     2571
      j=m(l)                                                               2572
      u=a(j)*v(j)+dot_product(wr,x(:,j))                                   2573
      if(abs(u) .gt. vp(j)*sa)goto 17181                                   2573
      at=0.0                                                               2573
      goto 17191                                                           2574
17181 continue                                                             2574
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/  (v(j)+vp(j)*o   2576 
     *mal)))
17191 continue                                                             2577
17171 continue                                                             2577
      if(at .eq. a(j))goto 17211                                           2577
      del=at-a(j)                                                          2577
      a(j)=at                                                              2577
      dli=max(dli,v(j)*del**2)                                             2578
      wr=wr-del*w*x(:,j)                                                   2578
      f=f+del*x(:,j)                                                       2579
17211 continue                                                             2580
17161 continue                                                             2581
17162 continue                                                             2581
      if(dli.lt.cthr)goto 17152                                            2581
      if(nlp .le. maxit)goto 17231                                         2581
      jerr=-ilm                                                            2581
      return                                                               2581
17231 continue                                                             2582
      goto 17151                                                           2583
17152 continue                                                             2583
      goto 17041                                                           2584
17042 continue                                                             2584
      if(nin.gt.nx)goto 17032                                              2585
      e=q*exp(sign(min(abs(f),fmax),f))                                    2586
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)                         2587
      if(jerr .eq. 0)goto 17251                                            2587
      jerr=jerr-ilm                                                        2587
      go to 12180                                                          2587
17251 continue                                                             2588
      ix=0                                                                 2589
17260 do 17261 j=1,nin                                                     2589
      k=m(j)                                                               2590
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 17261                           2590
      ix=1                                                                 2590
      goto 17262                                                           2590
17261 continue                                                             2591
17262 continue                                                             2591
      if(ix .ne. 0)goto 17281                                              2592
17290 do 17291 k=1,ni                                                      2592
      if(ixx(k).eq.1)goto 17291                                            2592
      if(ju(k).eq.0)goto 17291                                             2593
      ga(k)=abs(dot_product(wr,x(:,k)))                                    2594
      if(ga(k) .le. sa*vp(k))goto 17311                                    2594
      ixx(k)=1                                                             2594
      ix=1                                                                 2594
17311 continue                                                             2595
17291 continue                                                             2596
17292 continue                                                             2596
      if(ix.eq.1) go to 10880                                              2597
      goto 17032                                                           2598
17281 continue                                                             2599
      goto 17031                                                           2600
17032 continue                                                             2600
      if(nin .le. nx)goto 17331                                            2600
      jerr=-10000-ilm                                                      2600
      goto 16952                                                           2600
17331 continue                                                             2601
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))                               2601
      kin(ilm)=nin                                                         2602
      alm(ilm)=al                                                          2602
      lmu=ilm                                                              2603
      dev(ilm)=(risk(no,ni,nk,dq,dk,f,e,kp,jp,uu)-r0)/rr                   2604
      if(ilm.lt.mnl)goto 16951                                             2604
      if(flmin.ge.1.0)goto 16951                                           2605
      me=0                                                                 2605
17340 do 17341 j=1,nin                                                     2605
      if(ao(j,ilm).ne.0.0) me=me+1                                         2605
17341 continue                                                             2605
17342 continue                                                             2605
      if(me.gt.ne)goto 16952                                               2606
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 16952              2607
      if(dev(ilm).gt.devmax)goto 16952                                     2608
16951 continue                                                             2609
16952 continue                                                             2609
      g=f                                                                  2610
12180 continue                                                             2610
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx)              2611
      return                                                               2612
      end                                                                  2613
      subroutine cxmodval(ca,ia,nin,n,x,f)                                 2614
      real ca(nin),x(n,*),f(n)                                             2614
      integer ia(nin)                                                      2615
      f=0.0                                                                2615
      if(nin.le.0) return                                                  2616
17350 do 17351 i=1,n                                                       2616
      f(i)=f(i)+dot_product(ca(1:nin),x(i,ia(1:nin)))                      2616
17351 continue                                                             2617
17352 continue                                                             2617
      return                                                               2618
      end                                                                  2619
      subroutine groups(no,y,d,q,nk,kp,jp,t0,jerr)                         2620
      real y(no),d(no),q(no)                                               2620
      integer jp(no),kp(*)                                                 2621
17360 do 17361 j=1,no                                                      2621
      jp(j)=j                                                              2621
17361 continue                                                             2621
17362 continue                                                             2621
      call psort7(y,jp,1,no)                                               2622
      nj=0                                                                 2622
17370 do 17371 j=1,no                                                      2622
      if(q(jp(j)).le.0.0)goto 17371                                        2622
      nj=nj+1                                                              2622
      jp(nj)=jp(j)                                                         2622
17371 continue                                                             2623
17372 continue                                                             2623
      if(nj .ne. 0)goto 17391                                              2623
      jerr=20000                                                           2623
      return                                                               2623
17391 continue                                                             2624
      j=1                                                                  2624
17400 continue                                                             2624
17401 if(d(jp(j)).gt.0.0)goto 17402                                        2624
      j=j+1                                                                2624
      if(j.gt.nj)goto 17402                                                2624
      goto 17401                                                           2625
17402 continue                                                             2625
      if(j .lt. nj-1)goto 17421                                            2625
      jerr=30000                                                           2625
      return                                                               2625
17421 continue                                                             2626
      t0=y(jp(j))                                                          2626
      j0=j-1                                                               2627
      if(j0 .le. 0)goto 17441                                              2628
17450 continue                                                             2628
17451 if(y(jp(j0)).lt.t0)goto 17452                                        2628
      j0=j0-1                                                              2628
      if(j0.eq.0)goto 17452                                                2628
      goto 17451                                                           2629
17452 continue                                                             2629
      if(j0 .le. 0)goto 17471                                              2629
      nj=nj-j0                                                             2629
17480 do 17481 j=1,nj                                                      2629
      jp(j)=jp(j+j0)                                                       2629
17481 continue                                                             2629
17482 continue                                                             2629
17471 continue                                                             2630
17441 continue                                                             2631
      jerr=0                                                               2631
      nk=0                                                                 2631
      yk=t0                                                                2631
      j=2                                                                  2632
17490 continue                                                             2632
17491 continue                                                             2632
17500 continue                                                             2633
17501 if(d(jp(j)).gt.0.0.and.y(jp(j)).gt.yk)goto 17502                     2633
      j=j+1                                                                2633
      if(j.gt.nj)goto 17502                                                2633
      goto 17501                                                           2634
17502 continue                                                             2634
      nk=nk+1                                                              2634
      kp(nk)=j-1                                                           2634
      if(j.gt.nj)goto 17492                                                2635
      if(j .ne. nj)goto 17521                                              2635
      nk=nk+1                                                              2635
      kp(nk)=nj                                                            2635
      goto 17492                                                           2635
17521 continue                                                             2636
      yk=y(jp(j))                                                          2636
      j=j+1                                                                2637
      goto 17491                                                           2638
17492 continue                                                             2638
      return                                                               2639
      end                                                                  2640
      subroutine outer(no,nk,d,dk,kp,jp,e,wr,w,jerr,u)                     2641
      real d(no),dk(nk),wr(no),w(no)                                       2642
      real e(no),u(no),b,c                                                 2642
      integer kp(nk),jp(no)                                                2643
      call usk(no,nk,kp,jp,e,u)                                            2644
      b=dk(1)/u(1)                                                         2644
      c=dk(1)/u(1)**2                                                      2644
      jerr=0                                                               2645
17530 do 17531 j=1,kp(1)                                                   2645
      i=jp(j)                                                              2646
      w(i)=e(i)*(b-e(i)*c)                                                 2646
      if(w(i) .gt. 0.0)goto 17551                                          2646
      jerr=-30000                                                          2646
      return                                                               2646
17551 continue                                                             2647
      wr(i)=d(i)-e(i)*b                                                    2648
17531 continue                                                             2649
17532 continue                                                             2649
17560 do 17561 k=2,nk                                                      2649
      j1=kp(k-1)+1                                                         2649
      j2=kp(k)                                                             2650
      b=b+dk(k)/u(k)                                                       2650
      c=c+dk(k)/u(k)**2                                                    2651
17570 do 17571 j=j1,j2                                                     2651
      i=jp(j)                                                              2652
      w(i)=e(i)*(b-e(i)*c)                                                 2652
      if(w(i) .gt. 0.0)goto 17591                                          2652
      jerr=-30000                                                          2652
      return                                                               2652
17591 continue                                                             2653
      wr(i)=d(i)-e(i)*b                                                    2654
17571 continue                                                             2655
17572 continue                                                             2655
17561 continue                                                             2656
17562 continue                                                             2656
      return                                                               2657
      end                                                                  2658
      subroutine vars(no,ni,x,w,ixx,v)                                     2659
      real x(no,ni),w(no),v(ni)                                            2659
      integer ixx(ni)                                                      2660
17600 do 17601 j=1,ni                                                      2660
      if(ixx(j).gt.0) v(j)=dot_product(w,x(:,j)**2)                        2660
17601 continue                                                             2661
17602 continue                                                             2661
      return                                                               2662
      end                                                                  2663
      subroutine died(no,nk,d,kp,jp,dk)                                    2664
      real d(no),dk(nk)                                                    2664
      integer kp(nk),jp(no)                                                2665
      dk(1)=sum(d(jp(1:kp(1))))                                            2666
17610 do 17611 k=2,nk                                                      2666
      dk(k)=sum(d(jp((kp(k-1)+1):kp(k))))                                  2666
17611 continue                                                             2667
17612 continue                                                             2667
      return                                                               2668
      end                                                                  2669
      subroutine usk(no,nk,kp,jp,e,u)                                      2670
      real e(no),u(nk),h                                                   2670
      integer kp(nk),jp(no)                                                2671
      h=0.0                                                                2672
17620 do 17621 k=nk,1,-1                                                   2672
      j2=kp(k)                                                             2673
      j1=1                                                                 2673
      if(k.gt.1) j1=kp(k-1)+1                                              2674
17630 do 17631 j=j2,j1,-1                                                  2674
      h=h+e(jp(j))                                                         2674
17631 continue                                                             2675
17632 continue                                                             2675
      u(k)=h                                                               2676
17621 continue                                                             2677
17622 continue                                                             2677
      return                                                               2678
      end                                                                  2679
      function risk(no,ni,nk,d,dk,f,e,kp,jp,u)                             2680
      real d(no),dk(nk),f(no)                                              2681
      integer kp(nk),jp(no)                                                2681
      real e(no),u(nk),s                                                   2682
      call usk(no,nk,kp,jp,e,u)                                            2682
      u=log(u)                                                             2683
      risk=dot_product(d,f)-dot_product(dk,u)                              2684
      return                                                               2685
      end                                                                  2686
      subroutine loglike(no,ni,x,y,d,g,w,nlam,a,flog,jerr)                 2687
      real x(no,ni),y(no),d(no),g(no),w(no),a(ni,nlam),flog(nlam)          2688
      real, dimension (:), allocatable :: dk,f,xm,dq,q                          
      real, dimension (:), allocatable :: e,uu                                  
      integer, dimension (:), allocatable :: jp,kp                              
      allocate(e(1:no),stat=jerr)                                          2694
      allocate(q(1:no),stat=ierr)                                          2694
      jerr=jerr+ierr                                                       2695
      allocate(uu(1:no),stat=ierr)                                         2695
      jerr=jerr+ierr                                                       2696
      allocate(f(1:no),stat=ierr)                                          2696
      jerr=jerr+ierr                                                       2697
      allocate(dk(1:no),stat=ierr)                                         2697
      jerr=jerr+ierr                                                       2698
      allocate(jp(1:no),stat=ierr)                                         2698
      jerr=jerr+ierr                                                       2699
      allocate(kp(1:no),stat=ierr)                                         2699
      jerr=jerr+ierr                                                       2700
      allocate(dq(1:no),stat=ierr)                                         2700
      jerr=jerr+ierr                                                       2701
      allocate(xm(1:ni),stat=ierr)                                         2701
      jerr=jerr+ierr                                                       2702
      if(jerr.ne.0) go to 12180                                            2703
      q=max(0.0,w)                                                         2703
      sw=sum(q)                                                            2704
      if(sw .gt. 0.0)goto 17651                                            2704
      jerr=9999                                                            2704
      go to 12180                                                          2704
17651 continue                                                             2705
      call groups(no,y,d,q,nk,kp,jp,t0,jerr)                               2706
      if(jerr.ne.0) go to 12180                                            2706
      fmax=log(huge(e(1))*0.1)                                             2707
      dq=d*q                                                               2707
      call died(no,nk,dq,kp,jp,dk)                                         2707
      gm=dot_product(q,g)/sw                                               2708
17660 do 17661 j=1,ni                                                      2708
      xm(j)=dot_product(q,x(:,j))/sw                                       2708
17661 continue                                                             2709
17662 continue                                                             2709
17670 do 17671 lam=1,nlam                                                  2710
17680 do 17681 i=1,no                                                      2710
      f(i)=g(i)-gm+dot_product(a(:,lam),(x(i,:)-xm))                       2711
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))                        2712
17681 continue                                                             2713
17682 continue                                                             2713
      flog(lam)=risk(no,ni,nk,dq,dk,f,e,kp,jp,uu)                          2714
17671 continue                                                             2715
17672 continue                                                             2715
12180 continue                                                             2715
      deallocate(e,uu,dk,f,jp,kp,dq)                                       2716
      return                                                               2717
      end                                                                  2718
      subroutine fishnet (parm,no,ni,x,y,g,w,jd,vp,cl,ne,nx,nlam,flmin,u   2720 
     *lam,thr,  isd,intr,maxit,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      real x(no,ni),y(no),g(no),w(no),vp(ni),ulam(nlam)                    2721
      real ca(nx,nlam),a0(nlam),dev(nlam),alm(nlam),cl(2,ni)               2722
      integer jd(*),ia(nx),nin(nlam)                                       2723
      real, dimension (:), allocatable :: xm,xs,ww,vq                           
      integer, dimension (:), allocatable :: ju                                 
      if(maxval(vp) .gt. 0.0)goto 17701                                    2727
      jerr=10000                                                           2727
      return                                                               2727
17701 continue                                                             2728
      if(minval(y) .ge. 0.0)goto 17721                                     2728
      jerr=8888                                                            2728
      return                                                               2728
17721 continue                                                             2729
      allocate(ww(1:no),stat=jerr)                                         2730
      allocate(ju(1:ni),stat=ierr)                                         2730
      jerr=jerr+ierr                                                       2731
      allocate(vq(1:ni),stat=ierr)                                         2731
      jerr=jerr+ierr                                                       2732
      allocate(xm(1:ni),stat=ierr)                                         2732
      jerr=jerr+ierr                                                       2733
      if(isd .le. 0)goto 17741                                             2733
      allocate(xs(1:ni),stat=ierr)                                         2733
      jerr=jerr+ierr                                                       2733
17741 continue                                                             2734
      if(jerr.ne.0) return                                                 2735
      call chkvars(no,ni,x,ju)                                             2736
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 2737
      if(maxval(ju) .gt. 0)goto 17761                                      2737
      jerr=7777                                                            2737
      go to 12180                                                          2737
17761 continue                                                             2738
      vq=max(0.0,vp)                                                       2738
      vq=vq*ni/sum(vq)                                                     2739
      ww=max(0.0,w)                                                        2739
      sw=sum(ww)                                                           2739
      if(sw .gt. 0.0)goto 17781                                            2739
      jerr=9999                                                            2739
      go to 12180                                                          2739
17781 continue                                                             2740
      ww=ww/sw                                                             2741
      call lstandard1(no,ni,x,ww,ju,isd,intr,xm,xs)                        2742
      if(isd .le. 0)goto 17801                                             2742
17810 do 17811 j=1,ni                                                      2742
      cl(:,j)=cl(:,j)*xs(j)                                                2742
17811 continue                                                             2742
17812 continue                                                             2742
17801 continue                                                             2743
      call fishnet1(parm,no,ni,x,y,g,ww,ju,vq,cl,ne,nx,nlam,flmin,ulam,t   2745 
     *hr,  isd,intr,maxit,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) go to 12180                                            2745
      dev0=2.0*sw*dev0                                                     2746
17820 do 17821 k=1,lmu                                                     2746
      nk=nin(k)                                                            2747
      if(isd.gt.0) ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))                      2748
      if(intr .ne. 0)goto 17841                                            2748
      a0(k)=0.0                                                            2748
      goto 17851                                                           2749
17841 continue                                                             2749
      a0(k)=a0(k)-dot_product(ca(1:nk,k),xm(ia(1:nk)))                     2749
17851 continue                                                             2750
17831 continue                                                             2750
17821 continue                                                             2751
17822 continue                                                             2751
12180 continue                                                             2751
      deallocate(ww,ju,vq,xm)                                              2751
      if(isd.gt.0) deallocate(xs)                                          2752
      return                                                               2753
      end                                                                  2754
      subroutine fishnet1(parm,no,ni,x,y,g,q,ju,vp,cl,ne,nx,nlam,flmin,u   2756 
     *lam,shri,  isd,intr,maxit,lmu,a0,ca,m,kin,dev0,dev,alm,nlp,jerr)
      real x(no,ni),y(no),g(no),q(no),vp(ni),ulam(nlam)                    2757
      real ca(nx,nlam),a0(nlam),dev(nlam),alm(nlam),cl(2,ni)               2758
      integer ju(ni),m(nx),kin(nlam)                                       2759
      real, dimension (:), allocatable :: t,w,wr,v,a,f,as,ga                    
      integer, dimension (:), allocatable :: mm,ixx                             
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               2763
      sml=sml*10.0                                                         2764
      allocate(a(1:ni),stat=jerr)                                          2765
      allocate(as(1:ni),stat=ierr)                                         2765
      jerr=jerr+ierr                                                       2766
      allocate(t(1:no),stat=ierr)                                          2766
      jerr=jerr+ierr                                                       2767
      allocate(mm(1:ni),stat=ierr)                                         2767
      jerr=jerr+ierr                                                       2768
      allocate(ga(1:ni),stat=ierr)                                         2768
      jerr=jerr+ierr                                                       2769
      allocate(ixx(1:ni),stat=ierr)                                        2769
      jerr=jerr+ierr                                                       2770
      allocate(wr(1:no),stat=ierr)                                         2770
      jerr=jerr+ierr                                                       2771
      allocate(v(1:ni),stat=ierr)                                          2771
      jerr=jerr+ierr                                                       2772
      allocate(w(1:no),stat=ierr)                                          2772
      jerr=jerr+ierr                                                       2773
      allocate(f(1:no),stat=ierr)                                          2773
      jerr=jerr+ierr                                                       2774
      if(jerr.ne.0) return                                                 2775
      bta=parm                                                             2775
      omb=1.0-bta                                                          2776
      t=q*y                                                                2776
      yb=sum(t)                                                            2776
      fmax=log(huge(bta)*0.1)                                              2777
      if(nonzero(no,g) .ne. 0)goto 17871                                   2778
      if(intr .eq. 0)goto 17891                                            2778
      w=q*yb                                                               2778
      az=log(yb)                                                           2778
      f=az                                                                 2778
      dv0=yb*(az-1.0)                                                      2778
      goto 17901                                                           2779
17891 continue                                                             2779
      w=q                                                                  2779
      az=0.0                                                               2779
      f=az                                                                 2779
      dv0=-1.0                                                             2779
17901 continue                                                             2780
17881 continue                                                             2780
      goto 17911                                                           2781
17871 continue                                                             2781
      w=q*exp(sign(min(abs(g),fmax),g))                                    2781
      v0=sum(w)                                                            2782
      if(intr .eq. 0)goto 17931                                            2782
      eaz=yb/v0                                                            2782
      w=eaz*w                                                              2782
      az=log(eaz)                                                          2782
      f=az+g                                                               2783
      dv0=dot_product(t,g)-yb*(1.0-az)                                     2784
      goto 17941                                                           2785
17931 continue                                                             2785
      az=0.0                                                               2785
      f=g                                                                  2785
      dv0=dot_product(t,g)-v0                                              2785
17941 continue                                                             2786
17921 continue                                                             2786
17911 continue                                                             2787
17861 continue                                                             2787
      a=0.0                                                                2787
      as=0.0                                                               2787
      wr=t-w                                                               2787
      v0=1.0                                                               2787
      if(intr.ne.0) v0=yb                                                  2787
      dvr=-yb                                                              2788
17950 do 17951 i=1,no                                                      2788
      if(t(i).gt.0.0) dvr=dvr+t(i)*log(y(i))                               2788
17951 continue                                                             2788
17952 continue                                                             2788
      dvr=dvr-dv0                                                          2788
      dev0=dvr                                                             2789
      if(flmin .ge. 1.0)goto 17971                                         2789
      eqs=max(eps,flmin)                                                   2789
      alf=eqs**(1.0/(nlam-1))                                              2789
17971 continue                                                             2790
      m=0                                                                  2790
      mm=0                                                                 2790
      nlp=0                                                                2790
      nin=nlp                                                              2790
      mnl=min(mnlam,nlam)                                                  2790
      shr=shri*dev0                                                        2790
      ixx=0                                                                2790
      al=0.0                                                               2791
17980 do 17981 j=1,ni                                                      2791
      if(ju(j).eq.0)goto 17981                                             2791
      ga(j)=abs(dot_product(wr,x(:,j)))                                    2791
17981 continue                                                             2792
17982 continue                                                             2792
17990 do 17991 ilm=1,nlam                                                  2792
      al0=al                                                               2793
      if(flmin .lt. 1.0)goto 18011                                         2793
      al=ulam(ilm)                                                         2793
      goto 18001                                                           2794
18011 if(ilm .le. 2)goto 18021                                             2794
      al=al*alf                                                            2794
      goto 18001                                                           2795
18021 if(ilm .ne. 1)goto 18031                                             2795
      al=big                                                               2795
      goto 18041                                                           2796
18031 continue                                                             2796
      al0=0.0                                                              2797
18050 do 18051 j=1,ni                                                      2797
      if(ju(j).eq.0)goto 18051                                             2797
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            2797
18051 continue                                                             2798
18052 continue                                                             2798
      al0=al0/max(bta,1.0e-3)                                              2798
      al=alf*al0                                                           2799
18041 continue                                                             2800
18001 continue                                                             2800
      al2=al*omb                                                           2800
      al1=al*bta                                                           2800
      tlam=bta*(2.0*al-al0)                                                2801
18060 do 18061 k=1,ni                                                      2801
      if(ixx(k).eq.1)goto 18061                                            2801
      if(ju(k).eq.0)goto 18061                                             2802
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     2803
18061 continue                                                             2804
18062 continue                                                             2804
10880 continue                                                             2805
18070 continue                                                             2805
18071 continue                                                             2805
      az0=az                                                               2806
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))                                2807
18080 do 18081 j=1,ni                                                      2807
      if(ixx(j).ne.0) v(j)=dot_product(w,x(:,j)**2)                        2807
18081 continue                                                             2808
18082 continue                                                             2808
18090 continue                                                             2808
18091 continue                                                             2808
      nlp=nlp+1                                                            2808
      dlx=0.0                                                              2809
18100 do 18101 k=1,ni                                                      2809
      if(ixx(k).eq.0)goto 18101                                            2809
      ak=a(k)                                                              2810
      u=dot_product(wr,x(:,k))+v(k)*ak                                     2810
      au=abs(u)-vp(k)*al1                                                  2811
      if(au .gt. 0.0)goto 18121                                            2811
      a(k)=0.0                                                             2811
      goto 18131                                                           2812
18121 continue                                                             2813
      a(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(v(k)+vp(k)*al2)))           2814
18131 continue                                                             2815
18111 continue                                                             2815
      if(a(k).eq.ak)goto 18101                                             2815
      d=a(k)-ak                                                            2815
      dlx=max(dlx,v(k)*d**2)                                               2816
      wr=wr-d*w*x(:,k)                                                     2816
      f=f+d*x(:,k)                                                         2817
      if(mm(k) .ne. 0)goto 18151                                           2817
      nin=nin+1                                                            2817
      if(nin.gt.nx)goto 18102                                              2818
      mm(k)=nin                                                            2818
      m(nin)=k                                                             2819
18151 continue                                                             2820
18101 continue                                                             2821
18102 continue                                                             2821
      if(nin.gt.nx)goto 18092                                              2822
      if(intr .eq. 0)goto 18171                                            2822
      d=sum(wr)/v0                                                         2823
      az=az+d                                                              2823
      dlx=max(dlx,v0*d**2)                                                 2823
      wr=wr-d*w                                                            2823
      f=f+d                                                                2824
18171 continue                                                             2825
      if(dlx.lt.shr)goto 18092                                             2825
      if(nlp .le. maxit)goto 18191                                         2825
      jerr=-ilm                                                            2825
      return                                                               2825
18191 continue                                                             2826
18200 continue                                                             2826
18201 continue                                                             2826
      nlp=nlp+1                                                            2826
      dlx=0.0                                                              2827
18210 do 18211 l=1,nin                                                     2827
      k=m(l)                                                               2827
      ak=a(k)                                                              2828
      u=dot_product(wr,x(:,k))+v(k)*ak                                     2828
      au=abs(u)-vp(k)*al1                                                  2829
      if(au .gt. 0.0)goto 18231                                            2829
      a(k)=0.0                                                             2829
      goto 18241                                                           2830
18231 continue                                                             2831
      a(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(v(k)+vp(k)*al2)))           2832
18241 continue                                                             2833
18221 continue                                                             2833
      if(a(k).eq.ak)goto 18211                                             2833
      d=a(k)-ak                                                            2833
      dlx=max(dlx,v(k)*d**2)                                               2834
      wr=wr-d*w*x(:,k)                                                     2834
      f=f+d*x(:,k)                                                         2836
18211 continue                                                             2836
18212 continue                                                             2836
      if(intr .eq. 0)goto 18261                                            2836
      d=sum(wr)/v0                                                         2836
      az=az+d                                                              2837
      dlx=max(dlx,v0*d**2)                                                 2837
      wr=wr-d*w                                                            2837
      f=f+d                                                                2838
18261 continue                                                             2839
      if(dlx.lt.shr)goto 18202                                             2839
      if(nlp .le. maxit)goto 18281                                         2839
      jerr=-ilm                                                            2839
      return                                                               2839
18281 continue                                                             2840
      goto 18201                                                           2841
18202 continue                                                             2841
      goto 18091                                                           2842
18092 continue                                                             2842
      if(nin.gt.nx)goto 18072                                              2843
      w=q*exp(sign(min(abs(f),fmax),f))                                    2843
      v0=sum(w)                                                            2843
      wr=t-w                                                               2844
      if(v0*(az-az0)**2 .ge. shr)goto 18301                                2844
      ix=0                                                                 2845
18310 do 18311 j=1,nin                                                     2845
      k=m(j)                                                               2846
      if(v(k)*(a(k)-as(k))**2.lt.shr)goto 18311                            2846
      ix=1                                                                 2846
      goto 18312                                                           2847
18311 continue                                                             2848
18312 continue                                                             2848
      if(ix .ne. 0)goto 18331                                              2849
18340 do 18341 k=1,ni                                                      2849
      if(ixx(k).eq.1)goto 18341                                            2849
      if(ju(k).eq.0)goto 18341                                             2850
      ga(k)=abs(dot_product(wr,x(:,k)))                                    2851
      if(ga(k) .le. al1*vp(k))goto 18361                                   2851
      ixx(k)=1                                                             2851
      ix=1                                                                 2851
18361 continue                                                             2852
18341 continue                                                             2853
18342 continue                                                             2853
      if(ix.eq.1) go to 10880                                              2854
      goto 18072                                                           2855
18331 continue                                                             2856
18301 continue                                                             2857
      goto 18071                                                           2858
18072 continue                                                             2858
      if(nin .le. nx)goto 18381                                            2858
      jerr=-10000-ilm                                                      2858
      goto 17992                                                           2858
18381 continue                                                             2859
      if(nin.gt.0) ca(1:nin,ilm)=a(m(1:nin))                               2859
      kin(ilm)=nin                                                         2860
      a0(ilm)=az                                                           2860
      alm(ilm)=al                                                          2860
      lmu=ilm                                                              2861
      dev(ilm)=(dot_product(t,f)-v0-dv0)/dvr                               2862
      if(ilm.lt.mnl)goto 17991                                             2862
      if(flmin.ge.1.0)goto 17991                                           2863
      me=0                                                                 2863
18390 do 18391 j=1,nin                                                     2863
      if(ca(j,ilm).ne.0.0) me=me+1                                         2863
18391 continue                                                             2863
18392 continue                                                             2863
      if(me.gt.ne)goto 17992                                               2864
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 17992              2865
      if(dev(ilm).gt.devmax)goto 17992                                     2866
17991 continue                                                             2867
17992 continue                                                             2867
      g=f                                                                  2868
12180 continue                                                             2868
      deallocate(t,w,wr,v,a,f,as,mm,ga,ixx)                                2869
      return                                                               2870
      end                                                                  2871
      function nonzero(n,v)                                                2872
      real v(n)                                                            2873
      nonzero=0                                                            2873
18400 do 18401 i=1,n                                                       2873
      if(v(i) .eq. 0.0)goto 18421                                          2873
      nonzero=1                                                            2873
      return                                                               2873
18421 continue                                                             2873
18401 continue                                                             2874
18402 continue                                                             2874
      return                                                               2875
      end                                                                  2876
      subroutine solns(ni,nx,lmu,a,ia,nin,b)                               2877
      real a(nx,lmu),b(ni,lmu)                                             2877
      integer ia(nx),nin(lmu)                                              2878
18430 do 18431 lam=1,lmu                                                   2878
      call uncomp(ni,a(:,lam),ia,nin(lam),b(:,lam))                        2878
18431 continue                                                             2879
18432 continue                                                             2879
      return                                                               2880
      end                                                                  2881
      subroutine lsolns(ni,nx,nc,lmu,a,ia,nin,b)                           2882
      real a(nx,nc,lmu),b(ni,nc,lmu)                                       2882
      integer ia(nx),nin(lmu)                                              2883
18440 do 18441 lam=1,lmu                                                   2883
      call luncomp(ni,nx,nc,a(1,1,lam),ia,nin(lam),b(1,1,lam))             2883
18441 continue                                                             2884
18442 continue                                                             2884
      return                                                               2885
      end                                                                  2886
      subroutine deviance(no,ni,x,y,g,q,nlam,a0,a,flog,jerr)               2887
      real x(no,ni),y(no),g(no),q(no),a(ni,nlam),a0(nlam),flog(nlam)       2888
      real, dimension (:), allocatable :: w                                     
      if(minval(y) .ge. 0.0)goto 18461                                     2891
      jerr=8888                                                            2891
      return                                                               2891
18461 continue                                                             2892
      allocate(w(1:no),stat=jerr)                                          2892
      if(jerr.ne.0) return                                                 2893
      w=max(0.0,q)                                                         2893
      sw=sum(w)                                                            2893
      if(sw .gt. 0.0)goto 18481                                            2893
      jerr=9999                                                            2893
      go to 12180                                                          2893
18481 continue                                                             2894
      yb=dot_product(w,y)/sw                                               2894
      fmax=log(huge(y(1))*0.1)                                             2895
18490 do 18491 lam=1,nlam                                                  2895
      s=0.0                                                                2896
18500 do 18501 i=1,no                                                      2896
      if(w(i).le.0.0)goto 18501                                            2897
      f=g(i)+a0(lam)+dot_product(a(:,lam),x(i,:))                          2898
      s=s+w(i)*(y(i)*f-exp(sign(min(abs(f),fmax),f)))                      2899
18501 continue                                                             2900
18502 continue                                                             2900
      flog(lam)=2.0*(sw*yb*(log(yb)-1.0)-s)                                2901
18491 continue                                                             2902
18492 continue                                                             2902
12180 continue                                                             2902
      deallocate(w)                                                        2903
      return                                                               2904
      end                                                                  2905
      subroutine spfishnet (parm,no,ni,x,ix,jx,y,g,w,jd,vp,cl,ne,nx,nlam   2907 
     *,flmin,  ulam,thr,isd,intr,maxit,lmu,a0,ca,ia,nin,dev0,dev,alm,nlp
     *,jerr)
      real x(*),y(no),g(no),w(no),vp(ni),ulam(nlam),cl(2,ni)               2908
      real ca(nx,nlam),a0(nlam),dev(nlam),alm(nlam)                        2909
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           2910
      real, dimension (:), allocatable :: xm,xs,ww,vq                           
      integer, dimension (:), allocatable :: ju                                 
      if(maxval(vp) .gt. 0.0)goto 18521                                    2914
      jerr=10000                                                           2914
      return                                                               2914
18521 continue                                                             2915
      if(minval(y) .ge. 0.0)goto 18541                                     2915
      jerr=8888                                                            2915
      return                                                               2915
18541 continue                                                             2916
      allocate(ww(1:no),stat=jerr)                                         2917
      allocate(ju(1:ni),stat=ierr)                                         2917
      jerr=jerr+ierr                                                       2918
      allocate(vq(1:ni),stat=ierr)                                         2918
      jerr=jerr+ierr                                                       2919
      allocate(xm(1:ni),stat=ierr)                                         2919
      jerr=jerr+ierr                                                       2920
      allocate(xs(1:ni),stat=ierr)                                         2920
      jerr=jerr+ierr                                                       2921
      if(jerr.ne.0) return                                                 2922
      call spchkvars(no,ni,x,ix,ju)                                        2923
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 2924
      if(maxval(ju) .gt. 0)goto 18561                                      2924
      jerr=7777                                                            2924
      go to 12180                                                          2924
18561 continue                                                             2925
      vq=max(0.0,vp)                                                       2925
      vq=vq*ni/sum(vq)                                                     2926
      ww=max(0.0,w)                                                        2926
      sw=sum(ww)                                                           2926
      if(sw .gt. 0.0)goto 18581                                            2926
      jerr=9999                                                            2926
      go to 12180                                                          2926
18581 continue                                                             2927
      ww=ww/sw                                                             2928
      call splstandard2(no,ni,x,ix,jx,ww,ju,isd,intr,xm,xs)                2929
      if(isd .le. 0)goto 18601                                             2929
18610 do 18611 j=1,ni                                                      2929
      cl(:,j)=cl(:,j)*xs(j)                                                2929
18611 continue                                                             2929
18612 continue                                                             2929
18601 continue                                                             2930
      call spfishnet1(parm,no,ni,x,ix,jx,y,g,ww,ju,vq,cl,ne,nx,nlam,flmi   2932 
     *n,ulam,thr,  isd,intr,maxit,xm,xs,lmu,a0,ca,ia,nin,dev0,dev,alm,nl
     *p,jerr)
      if(jerr.gt.0) go to 12180                                            2932
      dev0=2.0*sw*dev0                                                     2933
18620 do 18621 k=1,lmu                                                     2933
      nk=nin(k)                                                            2934
      if(isd.gt.0) ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))                      2935
      if(intr .ne. 0)goto 18641                                            2935
      a0(k)=0.0                                                            2935
      goto 18651                                                           2936
18641 continue                                                             2936
      a0(k)=a0(k)-dot_product(ca(1:nk,k),xm(ia(1:nk)))                     2936
18651 continue                                                             2937
18631 continue                                                             2937
18621 continue                                                             2938
18622 continue                                                             2938
12180 continue                                                             2938
      deallocate(ww,ju,vq,xm,xs)                                           2939
      return                                                               2940
      end                                                                  2941
      subroutine spfishnet1(parm,no,ni,x,ix,jx,y,g,q,ju,vp,cl,ne,nx,nlam   2943 
     *,flmin,ulam,  shri,isd,intr,maxit,xb,xs,lmu,a0,ca,m,kin,dev0,dev,a
     *lm,nlp,jerr)
      real x(*),y(no),g(no),q(no),vp(ni),ulam(nlam),xb(ni),xs(ni)          2944
      real ca(nx,nlam),a0(nlam),dev(nlam),alm(nlam),cl(2,ni)               2945
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)                           2946
      real, dimension (:), allocatable :: qy,t,w,wr,v,a,as,xm,ga                
      integer, dimension (:), allocatable :: mm,ixx                             
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               2950
      sml=sml*10.0                                                         2951
      allocate(a(1:ni),stat=jerr)                                          2952
      allocate(as(1:ni),stat=ierr)                                         2952
      jerr=jerr+ierr                                                       2953
      allocate(t(1:no),stat=ierr)                                          2953
      jerr=jerr+ierr                                                       2954
      allocate(mm(1:ni),stat=ierr)                                         2954
      jerr=jerr+ierr                                                       2955
      allocate(ga(1:ni),stat=ierr)                                         2955
      jerr=jerr+ierr                                                       2956
      allocate(ixx(1:ni),stat=ierr)                                        2956
      jerr=jerr+ierr                                                       2957
      allocate(wr(1:no),stat=ierr)                                         2957
      jerr=jerr+ierr                                                       2958
      allocate(v(1:ni),stat=ierr)                                          2958
      jerr=jerr+ierr                                                       2959
      allocate(xm(1:ni),stat=ierr)                                         2959
      jerr=jerr+ierr                                                       2960
      allocate(w(1:no),stat=ierr)                                          2960
      jerr=jerr+ierr                                                       2961
      allocate(qy(1:no),stat=ierr)                                         2961
      jerr=jerr+ierr                                                       2962
      if(jerr.ne.0) return                                                 2963
      bta=parm                                                             2963
      omb=1.0-bta                                                          2963
      fmax=log(huge(bta)*0.1)                                              2964
      qy=q*y                                                               2964
      yb=sum(qy)                                                           2965
      if(nonzero(no,g) .ne. 0)goto 18671                                   2965
      t=0.0                                                                2966
      if(intr .eq. 0)goto 18691                                            2966
      w=q*yb                                                               2966
      az=log(yb)                                                           2966
      uu=az                                                                2967
      xm=yb*xb                                                             2967
      dv0=yb*(az-1.0)                                                      2968
      goto 18701                                                           2969
18691 continue                                                             2969
      w=q                                                                  2969
      xm=0.0                                                               2969
      uu=0.0                                                               2969
      az=uu                                                                2969
      dv0=-1.0                                                             2969
18701 continue                                                             2970
18681 continue                                                             2970
      goto 18711                                                           2971
18671 continue                                                             2971
      w=q*exp(sign(min(abs(g),fmax),g))                                    2971
      ww=sum(w)                                                            2971
      t=g                                                                  2972
      if(intr .eq. 0)goto 18731                                            2972
      eaz=yb/ww                                                            2973
      w=eaz*w                                                              2973
      az=log(eaz)                                                          2973
      uu=az                                                                2973
      dv0=dot_product(qy,g)-yb*(1.0-az)                                    2974
      goto 18741                                                           2975
18731 continue                                                             2975
      uu=0.0                                                               2975
      az=uu                                                                2975
      dv0=dot_product(qy,g)-ww                                             2975
18741 continue                                                             2976
18721 continue                                                             2976
18750 do 18751 j=1,ni                                                      2976
      if(ju(j).eq.0)goto 18751                                             2976
      jb=ix(j)                                                             2976
      je=ix(j+1)-1                                                         2977
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             2978
18751 continue                                                             2979
18752 continue                                                             2979
18711 continue                                                             2980
18661 continue                                                             2980
      tt=yb*uu                                                             2980
      ww=1.0                                                               2980
      if(intr.ne.0) ww=yb                                                  2980
      wr=qy-q*(yb*(1.0-uu))                                                2980
      a=0.0                                                                2980
      as=0.0                                                               2981
      dvr=-yb                                                              2982
18760 do 18761 i=1,no                                                      2982
      if(qy(i).gt.0.0) dvr=dvr+qy(i)*log(y(i))                             2982
18761 continue                                                             2982
18762 continue                                                             2982
      dvr=dvr-dv0                                                          2982
      dev0=dvr                                                             2983
      if(flmin .ge. 1.0)goto 18781                                         2983
      eqs=max(eps,flmin)                                                   2983
      alf=eqs**(1.0/(nlam-1))                                              2983
18781 continue                                                             2984
      m=0                                                                  2984
      mm=0                                                                 2984
      nlp=0                                                                2984
      nin=nlp                                                              2984
      mnl=min(mnlam,nlam)                                                  2984
      shr=shri*dev0                                                        2984
      al=0.0                                                               2984
      ixx=0                                                                2985
18790 do 18791 j=1,ni                                                      2985
      if(ju(j).eq.0)goto 18791                                             2986
      jb=ix(j)                                                             2986
      je=ix(j+1)-1                                                         2987
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))  -uu*(xm(j)-ww*xb(j)   2989 
     *)-xb(j)*tt)/xs(j)
18791 continue                                                             2990
18792 continue                                                             2990
18800 do 18801 ilm=1,nlam                                                  2990
      al0=al                                                               2991
      if(flmin .lt. 1.0)goto 18821                                         2991
      al=ulam(ilm)                                                         2991
      goto 18811                                                           2992
18821 if(ilm .le. 2)goto 18831                                             2992
      al=al*alf                                                            2992
      goto 18811                                                           2993
18831 if(ilm .ne. 1)goto 18841                                             2993
      al=big                                                               2993
      goto 18851                                                           2994
18841 continue                                                             2994
      al0=0.0                                                              2995
18860 do 18861 j=1,ni                                                      2995
      if(ju(j).eq.0)goto 18861                                             2995
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            2995
18861 continue                                                             2996
18862 continue                                                             2996
      al0=al0/max(bta,1.0e-3)                                              2996
      al=alf*al0                                                           2997
18851 continue                                                             2998
18811 continue                                                             2998
      al2=al*omb                                                           2998
      al1=al*bta                                                           2998
      tlam=bta*(2.0*al-al0)                                                2999
18870 do 18871 k=1,ni                                                      2999
      if(ixx(k).eq.1)goto 18871                                            2999
      if(ju(k).eq.0)goto 18871                                             3000
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     3001
18871 continue                                                             3002
18872 continue                                                             3002
10880 continue                                                             3003
18880 continue                                                             3003
18881 continue                                                             3003
      az0=az                                                               3004
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))                                3005
18890 do 18891 j=1,ni                                                      3005
      if(ixx(j).eq.0)goto 18891                                            3005
      jb=ix(j)                                                             3005
      je=ix(j+1)-1                                                         3006
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             3007
      v(j)=(dot_product(w(jx(jb:je)),x(jb:je)**2)  -2.0*xb(j)*xm(j)+ww*x   3009 
     *b(j)**2)/xs(j)**2
18891 continue                                                             3010
18892 continue                                                             3010
18900 continue                                                             3010
18901 continue                                                             3010
      nlp=nlp+1                                                            3011
      dlx=0.0                                                              3012
18910 do 18911 k=1,ni                                                      3012
      if(ixx(k).eq.0)goto 18911                                            3012
      jb=ix(k)                                                             3012
      je=ix(k+1)-1                                                         3012
      ak=a(k)                                                              3013
      u=(dot_product(wr(jx(jb:je)),x(jb:je))  -uu*(xm(k)-ww*xb(k))-xb(k)   3015 
     **tt)/xs(k)+v(k)*ak
      au=abs(u)-vp(k)*al1                                                  3016
      if(au .gt. 0.0)goto 18931                                            3016
      a(k)=0.0                                                             3016
      goto 18941                                                           3017
18931 continue                                                             3018
      a(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(v(k)+vp(k)*al2)))           3019
18941 continue                                                             3020
18921 continue                                                             3020
      if(a(k).eq.ak)goto 18911                                             3021
      if(mm(k) .ne. 0)goto 18961                                           3021
      nin=nin+1                                                            3021
      if(nin.gt.nx)goto 18912                                              3022
      mm(k)=nin                                                            3022
      m(nin)=k                                                             3023
18961 continue                                                             3024
      d=a(k)-ak                                                            3024
      dlx=max(dlx,v(k)*d**2)                                               3024
      dv=d/xs(k)                                                           3025
      wr(jx(jb:je))=wr(jx(jb:je))-dv*w(jx(jb:je))*x(jb:je)                 3026
      t(jx(jb:je))=t(jx(jb:je))+dv*x(jb:je)                                3027
      uu=uu-dv*xb(k)                                                       3027
      tt=tt-dv*xm(k)                                                       3028
18911 continue                                                             3029
18912 continue                                                             3029
      if(nin.gt.nx)goto 18902                                              3030
      if(intr .eq. 0)goto 18981                                            3030
      d=tt/ww-uu                                                           3031
      az=az+d                                                              3031
      dlx=max(dlx,ww*d**2)                                                 3031
      uu=uu+d                                                              3032
18981 continue                                                             3033
      if(dlx.lt.shr)goto 18902                                             3033
      if(nlp .le. maxit)goto 19001                                         3033
      jerr=-ilm                                                            3033
      return                                                               3033
19001 continue                                                             3034
19010 continue                                                             3034
19011 continue                                                             3034
      nlp=nlp+1                                                            3034
      dlx=0.0                                                              3035
19020 do 19021 l=1,nin                                                     3035
      k=m(l)                                                               3036
      jb=ix(k)                                                             3036
      je=ix(k+1)-1                                                         3036
      ak=a(k)                                                              3037
      u=(dot_product(wr(jx(jb:je)),x(jb:je))  -uu*(xm(k)-ww*xb(k))-xb(k)   3039 
     **tt)/xs(k)+v(k)*ak
      au=abs(u)-vp(k)*al1                                                  3040
      if(au .gt. 0.0)goto 19041                                            3040
      a(k)=0.0                                                             3040
      goto 19051                                                           3041
19041 continue                                                             3042
      a(k)=max(cl(1,k),min(cl(2,k),sign(au,u)/(v(k)+vp(k)*al2)))           3043
19051 continue                                                             3044
19031 continue                                                             3044
      if(a(k).eq.ak)goto 19021                                             3044
      d=a(k)-ak                                                            3044
      dlx=max(dlx,v(k)*d**2)                                               3045
      dv=d/xs(k)                                                           3045
      wr(jx(jb:je))=wr(jx(jb:je))-dv*w(jx(jb:je))*x(jb:je)                 3046
      t(jx(jb:je))=t(jx(jb:je))+dv*x(jb:je)                                3047
      uu=uu-dv*xb(k)                                                       3047
      tt=tt-dv*xm(k)                                                       3048
19021 continue                                                             3049
19022 continue                                                             3049
      if(intr .eq. 0)goto 19071                                            3049
      d=tt/ww-uu                                                           3049
      az=az+d                                                              3050
      dlx=max(dlx,ww*d**2)                                                 3050
      uu=uu+d                                                              3051
19071 continue                                                             3052
      if(dlx.lt.shr)goto 19012                                             3052
      if(nlp .le. maxit)goto 19091                                         3052
      jerr=-ilm                                                            3052
      return                                                               3052
19091 continue                                                             3053
      goto 19011                                                           3054
19012 continue                                                             3054
      goto 18901                                                           3055
18902 continue                                                             3055
      if(nin.gt.nx)goto 18882                                              3056
      euu=exp(sign(min(abs(uu),fmax),uu))                                  3057
      w=euu*q*exp(sign(min(abs(t),fmax),t))                                3057
      ww=sum(w)                                                            3058
      wr=qy-w*(1.0-uu)                                                     3058
      tt=sum(wr)                                                           3059
      if(ww*(az-az0)**2 .ge. shr)goto 19111                                3059
      kx=0                                                                 3060
19120 do 19121 j=1,nin                                                     3060
      k=m(j)                                                               3061
      if(v(k)*(a(k)-as(k))**2.lt.shr)goto 19121                            3061
      kx=1                                                                 3061
      goto 19122                                                           3062
19121 continue                                                             3063
19122 continue                                                             3063
      if(kx .ne. 0)goto 19141                                              3064
19150 do 19151 j=1,ni                                                      3064
      if(ixx(j).eq.1)goto 19151                                            3064
      if(ju(j).eq.0)goto 19151                                             3065
      jb=ix(j)                                                             3065
      je=ix(j+1)-1                                                         3066
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             3067
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))  -uu*(xm(j)-ww*xb(j)   3069 
     *)-xb(j)*tt)/xs(j)
      if(ga(j) .le. al1*vp(j))goto 19171                                   3069
      ixx(j)=1                                                             3069
      kx=1                                                                 3069
19171 continue                                                             3070
19151 continue                                                             3071
19152 continue                                                             3071
      if(kx.eq.1) go to 10880                                              3072
      goto 18882                                                           3073
19141 continue                                                             3074
19111 continue                                                             3075
      goto 18881                                                           3076
18882 continue                                                             3076
      if(nin .le. nx)goto 19191                                            3076
      jerr=-10000-ilm                                                      3076
      goto 18802                                                           3076
19191 continue                                                             3077
      if(nin.gt.0) ca(1:nin,ilm)=a(m(1:nin))                               3077
      kin(ilm)=nin                                                         3078
      a0(ilm)=az                                                           3078
      alm(ilm)=al                                                          3078
      lmu=ilm                                                              3079
      dev(ilm)=(dot_product(qy,t)+yb*uu-ww-dv0)/dvr                        3080
      if(ilm.lt.mnl)goto 18801                                             3080
      if(flmin.ge.1.0)goto 18801                                           3081
      me=0                                                                 3081
19200 do 19201 j=1,nin                                                     3081
      if(ca(j,ilm).ne.0.0) me=me+1                                         3081
19201 continue                                                             3081
19202 continue                                                             3081
      if(me.gt.ne)goto 18802                                               3082
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 18802              3083
      if(dev(ilm).gt.devmax)goto 18802                                     3084
18801 continue                                                             3085
18802 continue                                                             3085
      g=t+uu                                                               3086
12180 continue                                                             3086
      deallocate(t,w,wr,v,a,qy,xm,as,mm,ga,ixx)                            3087
      return                                                               3088
      end                                                                  3089
      subroutine spdeviance(no,ni,x,ix,jx,y,g,q,nlam,a0,a,flog,jerr)       3090
      real x(*),y(no),g(no),q(no),a(ni,nlam),a0(nlam),flog(nlam)           3091
      integer ix(*),jx(*)                                                  3092
      real, dimension (:), allocatable :: w,f                                   
      if(minval(y) .ge. 0.0)goto 19221                                     3095
      jerr=8888                                                            3095
      return                                                               3095
19221 continue                                                             3096
      allocate(w(1:no),stat=jerr)                                          3097
      allocate(f(1:no),stat=ierr)                                          3097
      jerr=jerr+ierr                                                       3098
      if(jerr.ne.0) return                                                 3099
      w=max(0.0,q)                                                         3099
      sw=sum(w)                                                            3099
      if(sw .gt. 0.0)goto 19241                                            3099
      jerr=9999                                                            3099
      go to 12180                                                          3099
19241 continue                                                             3100
      yb=dot_product(w,y)/sw                                               3100
      fmax=log(huge(y(1))*0.1)                                             3101
19250 do 19251 lam=1,nlam                                                  3101
      f=a0(lam)                                                            3102
19260 do 19261 j=1,ni                                                      3102
      if(a(j,lam).eq.0.0)goto 19261                                        3102
      jb=ix(j)                                                             3102
      je=ix(j+1)-1                                                         3103
      f(jx(jb:je))=f(jx(jb:je))+a(j,lam)*x(jb:je)                          3104
19261 continue                                                             3105
19262 continue                                                             3105
      f=f+g                                                                3106
      s=dot_product(w,y*f-exp(sign(min(abs(f),fmax),f)))                   3107
      flog(lam)=2.0*(sw*yb*(log(yb)-1.0)-s)                                3108
19251 continue                                                             3109
19252 continue                                                             3109
12180 continue                                                             3109
      deallocate(w,f)                                                      3110
      return                                                               3111
      end                                                                  3112
      subroutine cspdeviance(no,x,ix,jx,y,g,q,nx,nlam,a0,ca,ia,nin,flog,   3113 
     *jerr)
      real x(*),y(no),g(no),q(no),ca(nx,nlam),a0(nlam),flog(nlam)          3114
      integer ix(*),jx(*),nin(nlam),ia(nx)                                 3115
      real, dimension (:), allocatable :: w,f                                   
      if(minval(y) .ge. 0.0)goto 19281                                     3118
      jerr=8888                                                            3118
      return                                                               3118
19281 continue                                                             3119
      allocate(w(1:no),stat=jerr)                                          3120
      allocate(f(1:no),stat=ierr)                                          3120
      jerr=jerr+ierr                                                       3121
      if(jerr.ne.0) return                                                 3122
      w=max(0.0,q)                                                         3122
      sw=sum(w)                                                            3122
      if(sw .gt. 0.0)goto 19301                                            3122
      jerr=9999                                                            3122
      go to 12180                                                          3122
19301 continue                                                             3123
      yb=dot_product(w,y)/sw                                               3123
      fmax=log(huge(y(1))*0.1)                                             3124
19310 do 19311 lam=1,nlam                                                  3124
      f=a0(lam)                                                            3125
19320 do 19321 k=1,nin(lam)                                                3125
      j=ia(k)                                                              3125
      jb=ix(j)                                                             3125
      je=ix(j+1)-1                                                         3126
      f(jx(jb:je))=f(jx(jb:je))+ca(k,lam)*x(jb:je)                         3127
19321 continue                                                             3128
19322 continue                                                             3128
      f=f+g                                                                3129
      s=dot_product(w,y*f-exp(sign(min(abs(f),fmax),f)))                   3130
      flog(lam)=2.0*(sw*yb*(log(yb)-1.0)-s)                                3131
19311 continue                                                             3132
19312 continue                                                             3132
12180 continue                                                             3132
      deallocate(w,f)                                                      3133
      return                                                               3134
      end                                                                  3135
      subroutine multelnet  (parm,no,ni,nr,x,y,w,jd,vp,cl,ne,nx,nlam,flm   3138 
     *in,ulam,thr,isd,jsd,intr,maxit,  lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr
     *)
      real x(no,ni),y(no,nr),w(no),vp(ni),ca(nx,nr,nlam)                   3139
      real ulam(nlam),a0(nr,nlam),rsq(nlam),alm(nlam),cl(2,ni)             3140
      integer jd(*),ia(nx),nin(nlam)                                       3141
      real, dimension (:), allocatable :: vq;                                   
      if(maxval(vp) .gt. 0.0)goto 19341                                    3144
      jerr=10000                                                           3144
      return                                                               3144
19341 continue                                                             3145
      allocate(vq(1:ni),stat=jerr)                                         3145
      if(jerr.ne.0) return                                                 3146
      vq=max(0.0,vp)                                                       3146
      vq=vq*ni/sum(vq)                                                     3147
      call multelnetn(parm,no,ni,nr,x,y,w,jd,vq,cl,ne,nx,nlam,flmin,ulam   3149 
     *,thr,isd,  jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr)
      deallocate(vq)                                                       3150
      return                                                               3151
      end                                                                  3152
      subroutine multelnetn (parm,no,ni,nr,x,y,w,jd,vp,cl,ne,nx,nlam,flm   3154 
     *in,ulam,thr,  isd,jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr
     *)
      real vp(ni),x(no,ni),y(no,nr),w(no),ulam(nlam),cl(2,ni)              3155
      real ca(nx,nr,nlam),a0(nr,nlam),rsq(nlam),alm(nlam)                  3156
      integer jd(*),ia(nx),nin(nlam)                                       3157
      real, dimension (:), allocatable :: xm,xs,xv,ym,ys                        
      integer, dimension (:), allocatable :: ju                                 
      real, dimension (:,:,:), allocatable :: clt                               
      allocate(clt(1:2,1:nr,1:ni),stat=jerr);                                   
      allocate(xm(1:ni),stat=ierr)                                         3163
      jerr=jerr+ierr                                                       3164
      allocate(xs(1:ni),stat=ierr)                                         3164
      jerr=jerr+ierr                                                       3165
      allocate(ym(1:nr),stat=ierr)                                         3165
      jerr=jerr+ierr                                                       3166
      allocate(ys(1:nr),stat=ierr)                                         3166
      jerr=jerr+ierr                                                       3167
      allocate(ju(1:ni),stat=ierr)                                         3167
      jerr=jerr+ierr                                                       3168
      allocate(xv(1:ni),stat=ierr)                                         3168
      jerr=jerr+ierr                                                       3169
      if(jerr.ne.0) return                                                 3170
      call chkvars(no,ni,x,ju)                                             3171
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 3172
      if(maxval(ju) .gt. 0)goto 19361                                      3172
      jerr=7777                                                            3172
      return                                                               3172
19361 continue                                                             3173
      call multstandard1(no,ni,nr,x,y,w,isd,jsd,intr,ju,xm,xs,ym,ys,xv,y   3174 
     *s0,jerr)
      if(jerr.ne.0) return                                                 3175
19370 do 19371 j=1,ni                                                      3175
19380 do 19381 k=1,nr                                                      3175
19390 do 19391 i=1,2                                                       3175
      clt(i,k,j)=cl(i,j)                                                   3175
19391 continue                                                             3175
19392 continue                                                             3175
19381 continue                                                             3175
19382 continue                                                             3175
19371 continue                                                             3176
19372 continue                                                             3176
      if(isd .le. 0)goto 19411                                             3176
19420 do 19421 j=1,ni                                                      3176
19430 do 19431 k=1,nr                                                      3176
19440 do 19441 i=1,2                                                       3176
      clt(i,k,j)=clt(i,k,j)*xs(j)                                          3176
19441 continue                                                             3176
19442 continue                                                             3176
19431 continue                                                             3176
19432 continue                                                             3176
19421 continue                                                             3176
19422 continue                                                             3176
19411 continue                                                             3177
      if(jsd .le. 0)goto 19461                                             3177
19470 do 19471 j=1,ni                                                      3177
19480 do 19481 k=1,nr                                                      3177
19490 do 19491 i=1,2                                                       3177
      clt(i,k,j)=clt(i,k,j)/ys(k)                                          3177
19491 continue                                                             3177
19492 continue                                                             3177
19481 continue                                                             3177
19482 continue                                                             3177
19471 continue                                                             3177
19472 continue                                                             3177
19461 continue                                                             3178
      call multelnet2(parm,ni,nr,ju,vp,clt,y,no,ne,nx,x,nlam,flmin,ulam,   3180 
     *thr,maxit,xv,  ys0,lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                 3181
19500 do 19501 k=1,lmu                                                     3181
      nk=nin(k)                                                            3182
19510 do 19511 j=1,nr                                                      3183
19520 do 19521 l=1,nk                                                      3183
      ca(l,j,k)=ys(j)*ca(l,j,k)/xs(ia(l))                                  3183
19521 continue                                                             3184
19522 continue                                                             3184
      if(intr .ne. 0)goto 19541                                            3184
      a0(j,k)=0.0                                                          3184
      goto 19551                                                           3185
19541 continue                                                             3185
      a0(j,k)=ym(j)-dot_product(ca(1:nk,j,k),xm(ia(1:nk)))                 3185
19551 continue                                                             3186
19531 continue                                                             3186
19511 continue                                                             3187
19512 continue                                                             3187
19501 continue                                                             3188
19502 continue                                                             3188
      deallocate(xm,xs,ym,ys,ju,xv,clt)                                    3189
      return                                                               3190
      end                                                                  3191
      subroutine multstandard1  (no,ni,nr,x,y,w,isd,jsd,intr,ju,xm,xs,ym   3193 
     *,ys,xv,ys0,jerr)
      real x(no,ni),y(no,nr),w(no),xm(ni),xs(ni),xv(ni),ym(nr),ys(nr)      3194
      integer ju(ni)                                                       3195
      real, dimension (:), allocatable :: v                                     
      allocate(v(1:no),stat=jerr)                                          3198
      if(jerr.ne.0) return                                                 3199
      w=w/sum(w)                                                           3199
      v=sqrt(w)                                                            3200
      if(intr .ne. 0)goto 19571                                            3201
19580 do 19581 j=1,ni                                                      3201
      if(ju(j).eq.0)goto 19581                                             3201
      xm(j)=0.0                                                            3201
      x(:,j)=v*x(:,j)                                                      3202
      z=dot_product(x(:,j),x(:,j))                                         3203
      if(isd .le. 0)goto 19601                                             3203
      xbq=dot_product(v,x(:,j))**2                                         3203
      vc=z-xbq                                                             3204
      xs(j)=sqrt(vc)                                                       3204
      x(:,j)=x(:,j)/xs(j)                                                  3204
      xv(j)=1.0+xbq/vc                                                     3205
      goto 19611                                                           3206
19601 continue                                                             3206
      xs(j)=1.0                                                            3206
      xv(j)=z                                                              3206
19611 continue                                                             3207
19591 continue                                                             3207
19581 continue                                                             3208
19582 continue                                                             3208
      ys0=0.0                                                              3209
19620 do 19621 j=1,nr                                                      3209
      ym(j)=0.0                                                            3209
      y(:,j)=v*y(:,j)                                                      3210
      z=dot_product(y(:,j),y(:,j))                                         3211
      if(jsd .le. 0)goto 19641                                             3211
      u=z-dot_product(v,y(:,j))**2                                         3211
      ys0=ys0+z/u                                                          3212
      ys(j)=sqrt(u)                                                        3212
      y(:,j)=y(:,j)/ys(j)                                                  3213
      goto 19651                                                           3214
19641 continue                                                             3214
      ys(j)=1.0                                                            3214
      ys0=ys0+z                                                            3214
19651 continue                                                             3215
19631 continue                                                             3215
19621 continue                                                             3216
19622 continue                                                             3216
      go to 10700                                                          3217
19571 continue                                                             3218
19660 do 19661 j=1,ni                                                      3218
      if(ju(j).eq.0)goto 19661                                             3219
      xm(j)=dot_product(w,x(:,j))                                          3219
      x(:,j)=v*(x(:,j)-xm(j))                                              3220
      xv(j)=dot_product(x(:,j),x(:,j))                                     3220
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                       3221
19661 continue                                                             3222
19662 continue                                                             3222
      if(isd .ne. 0)goto 19681                                             3222
      xs=1.0                                                               3222
      goto 19691                                                           3223
19681 continue                                                             3223
19700 do 19701 j=1,ni                                                      3223
      if(ju(j).eq.0)goto 19701                                             3223
      x(:,j)=x(:,j)/xs(j)                                                  3223
19701 continue                                                             3224
19702 continue                                                             3224
      xv=1.0                                                               3225
19691 continue                                                             3226
19671 continue                                                             3226
      ys0=0.0                                                              3227
19710 do 19711 j=1,nr                                                      3228
      ym(j)=dot_product(w,y(:,j))                                          3228
      y(:,j)=v*(y(:,j)-ym(j))                                              3229
      z=dot_product(y(:,j),y(:,j))                                         3230
      if(jsd .le. 0)goto 19731                                             3230
      ys(j)=sqrt(z)                                                        3230
      y(:,j)=y(:,j)/ys(j)                                                  3230
      goto 19741                                                           3231
19731 continue                                                             3231
      ys0=ys0+z                                                            3231
19741 continue                                                             3232
19721 continue                                                             3232
19711 continue                                                             3233
19712 continue                                                             3233
      if(jsd .ne. 0)goto 19761                                             3233
      ys=1.0                                                               3233
      goto 19771                                                           3233
19761 continue                                                             3233
      ys0=nr                                                               3233
19771 continue                                                             3234
19751 continue                                                             3234
10700 continue                                                             3234
      deallocate(v)                                                        3235
      return                                                               3236
      end                                                                  3237
      subroutine multelnet2(beta,ni,nr,ju,vp,cl,y,no,ne,nx,x,nlam,flmin,   3239 
     *ulam,thri,  maxit,xv,ys0,lmu,ao,ia,kin,rsqo,almo,nlp,jerr)
      real vp(ni),y(no,nr),x(no,ni),ulam(nlam),ao(nx,nr,nlam)              3240
      real rsqo(nlam),almo(nlam),xv(ni),cl(2,nr,ni)                        3241
      integer ju(ni),ia(nx),kin(nlam)                                      3242
      real, dimension (:), allocatable :: g,gk,del,gj                           
      integer, dimension (:), allocatable :: mm,ix,isc                          
      real, dimension (:,:), allocatable :: a                                   
      allocate(a(1:nr,1:ni),stat=jerr)                                          
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)               3249
      allocate(gj(1:nr),stat=ierr)                                         3249
      jerr=jerr+ierr                                                       3250
      allocate(gk(1:nr),stat=ierr)                                         3250
      jerr=jerr+ierr                                                       3251
      allocate(del(1:nr),stat=ierr)                                        3251
      jerr=jerr+ierr                                                       3252
      allocate(mm(1:ni),stat=ierr)                                         3252
      jerr=jerr+ierr                                                       3253
      allocate(g(1:ni),stat=ierr)                                          3253
      jerr=jerr+ierr                                                       3254
      allocate(ix(1:ni),stat=ierr)                                         3254
      jerr=jerr+ierr                                                       3255
      allocate(isc(1:nr),stat=ierr)                                        3255
      jerr=jerr+ierr                                                       3256
      if(jerr.ne.0) return                                                 3257
      bta=beta                                                             3257
      omb=1.0-bta                                                          3257
      ix=0                                                                 3257
      thr=thri*ys0/nr                                                      3258
      if(flmin .ge. 1.0)goto 19791                                         3258
      eqs=max(eps,flmin)                                                   3258
      alf=eqs**(1.0/(nlam-1))                                              3258
19791 continue                                                             3259
      rsq=ys0                                                              3259
      a=0.0                                                                3259
      mm=0                                                                 3259
      nlp=0                                                                3259
      nin=nlp                                                              3259
      iz=0                                                                 3259
      mnl=min(mnlam,nlam)                                                  3259
      alm=0.0                                                              3260
19800 do 19801 j=1,ni                                                      3260
      if(ju(j).eq.0)goto 19801                                             3260
      g(j)=0.0                                                             3261
19810 do 19811 k=1,nr                                                      3261
      g(j)=g(j)+dot_product(y(:,k),x(:,j))**2                              3261
19811 continue                                                             3262
19812 continue                                                             3262
      g(j)=sqrt(g(j))                                                      3263
19801 continue                                                             3264
19802 continue                                                             3264
19820 do 19821 m=1,nlam                                                    3264
      alm0=alm                                                             3265
      if(flmin .lt. 1.0)goto 19841                                         3265
      alm=ulam(m)                                                          3265
      goto 19831                                                           3266
19841 if(m .le. 2)goto 19851                                               3266
      alm=alm*alf                                                          3266
      goto 19831                                                           3267
19851 if(m .ne. 1)goto 19861                                               3267
      alm=big                                                              3267
      goto 19871                                                           3268
19861 continue                                                             3268
      alm0=0.0                                                             3269
19880 do 19881 j=1,ni                                                      3269
      if(ju(j).eq.0)goto 19881                                             3270
      if(vp(j).gt.0.0) alm0=max(alm0,g(j)/vp(j))                           3271
19881 continue                                                             3272
19882 continue                                                             3272
      alm0=alm0/max(bta,1.0e-3)                                            3272
      alm=alf*alm0                                                         3273
19871 continue                                                             3274
19831 continue                                                             3274
      dem=alm*omb                                                          3274
      ab=alm*bta                                                           3274
      rsq0=rsq                                                             3274
      jz=1                                                                 3275
      tlam=bta*(2.0*alm-alm0)                                              3276
19890 do 19891 k=1,ni                                                      3276
      if(ix(k).eq.1)goto 19891                                             3276
      if(ju(k).eq.0)goto 19891                                             3277
      if(g(k).gt.tlam*vp(k)) ix(k)=1                                       3278
19891 continue                                                             3279
19892 continue                                                             3279
19900 continue                                                             3279
19901 continue                                                             3279
      if(iz*jz.ne.0) go to 10360                                           3280
10880 continue                                                             3280
      nlp=nlp+1                                                            3280
      dlx=0.0                                                              3281
19910 do 19911 k=1,ni                                                      3281
      if(ix(k).eq.0)goto 19911                                             3281
      gkn=0.0                                                              3282
19920 do 19921 j=1,nr                                                      3282
      gj(j)=dot_product(y(:,j),x(:,k))                                     3283
      gk(j)=gj(j)+a(j,k)*xv(k)                                             3283
      gkn=gkn+gk(j)**2                                                     3285
19921 continue                                                             3285
19922 continue                                                             3285
      gkn=sqrt(gkn)                                                        3285
      u=1.0-ab*vp(k)/gkn                                                   3285
      del=a(:,k)                                                           3286
      if(u .gt. 0.0)goto 19941                                             3286
      a(:,k)=0.0                                                           3286
      goto 19951                                                           3287
19941 continue                                                             3287
      a(:,k)=gk*(u/(xv(k)+dem*vp(k)))                                      3288
      call chkbnds(nr,gk,gkn,xv(k),cl(1,1,k),  dem*vp(k),ab*vp(k),a(:,k)   3290 
     *,isc,jerr)
      if(jerr.ne.0) return                                                 3291
19951 continue                                                             3292
19931 continue                                                             3292
      del=a(:,k)-del                                                       3292
      if(maxval(abs(del)).le.0.0)goto 19911                                3293
19960 do 19961 j=1,nr                                                      3293
      rsq=rsq-del(j)*(2.0*gj(j)-del(j)*xv(k))                              3294
      y(:,j)=y(:,j)-del(j)*x(:,k)                                          3294
      dlx=max(dlx,xv(k)*del(j)**2)                                         3295
19961 continue                                                             3296
19962 continue                                                             3296
      if(mm(k) .ne. 0)goto 19981                                           3296
      nin=nin+1                                                            3296
      if(nin.gt.nx)goto 19912                                              3297
      mm(k)=nin                                                            3297
      ia(nin)=k                                                            3298
19981 continue                                                             3299
19911 continue                                                             3300
19912 continue                                                             3300
      if(nin.gt.nx)goto 19902                                              3301
      if(dlx .ge. thr)goto 20001                                           3301
      ixx=0                                                                3302
20010 do 20011 k=1,ni                                                      3302
      if(ix(k).eq.1)goto 20011                                             3302
      if(ju(k).eq.0)goto 20011                                             3302
      g(k)=0.0                                                             3303
20020 do 20021 j=1,nr                                                      3303
      g(k)=g(k)+dot_product(y(:,j),x(:,k))**2                              3303
20021 continue                                                             3304
20022 continue                                                             3304
      g(k)=sqrt(g(k))                                                      3305
      if(g(k) .le. ab*vp(k))goto 20041                                     3305
      ix(k)=1                                                              3305
      ixx=1                                                                3305
20041 continue                                                             3306
20011 continue                                                             3307
20012 continue                                                             3307
      if(ixx.eq.1) go to 10880                                             3308
      goto 19902                                                           3309
20001 continue                                                             3310
      if(nlp .le. maxit)goto 20061                                         3310
      jerr=-m                                                              3310
      return                                                               3310
20061 continue                                                             3311
10360 continue                                                             3311
      iz=1                                                                 3312
20070 continue                                                             3312
20071 continue                                                             3312
      nlp=nlp+1                                                            3312
      dlx=0.0                                                              3313
20080 do 20081 l=1,nin                                                     3313
      k=ia(l)                                                              3313
      gkn=0.0                                                              3314
20090 do 20091 j=1,nr                                                      3314
      gj(j)=dot_product(y(:,j),x(:,k))                                     3315
      gk(j)=gj(j)+a(j,k)*xv(k)                                             3315
      gkn=gkn+gk(j)**2                                                     3317
20091 continue                                                             3317
20092 continue                                                             3317
      gkn=sqrt(gkn)                                                        3317
      u=1.0-ab*vp(k)/gkn                                                   3317
      del=a(:,k)                                                           3318
      if(u .gt. 0.0)goto 20111                                             3318
      a(:,k)=0.0                                                           3318
      goto 20121                                                           3319
20111 continue                                                             3319
      a(:,k)=gk*(u/(xv(k)+dem*vp(k)))                                      3320
      call chkbnds(nr,gk,gkn,xv(k),cl(1,1,k),  dem*vp(k),ab*vp(k),a(:,k)   3322 
     *,isc,jerr)
      if(jerr.ne.0) return                                                 3323
20121 continue                                                             3324
20101 continue                                                             3324
      del=a(:,k)-del                                                       3324
      if(maxval(abs(del)).le.0.0)goto 20081                                3325
20130 do 20131 j=1,nr                                                      3325
      rsq=rsq-del(j)*(2.0*gj(j)-del(j)*xv(k))                              3326
      y(:,j)=y(:,j)-del(j)*x(:,k)                                          3326
      dlx=max(dlx,xv(k)*del(j)**2)                                         3327
20131 continue                                                             3328
20132 continue                                                             3328
20081 continue                                                             3329
20082 continue                                                             3329
      if(dlx.lt.thr)goto 20072                                             3329
      if(nlp .le. maxit)goto 20151                                         3329
      jerr=-m                                                              3329
      return                                                               3329
20151 continue                                                             3330
      goto 20071                                                           3331
20072 continue                                                             3331
      jz=0                                                                 3332
      goto 19901                                                           3333
19902 continue                                                             3333
      if(nin .le. nx)goto 20171                                            3333
      jerr=-10000-m                                                        3333
      goto 19822                                                           3333
20171 continue                                                             3334
      if(nin .le. 0)goto 20191                                             3334
20200 do 20201 j=1,nr                                                      3334
      ao(1:nin,j,m)=a(j,ia(1:nin))                                         3334
20201 continue                                                             3334
20202 continue                                                             3334
20191 continue                                                             3335
      kin(m)=nin                                                           3336
      rsqo(m)=1.0-rsq/ys0                                                  3336
      almo(m)=alm                                                          3336
      lmu=m                                                                3337
      if(m.lt.mnl)goto 19821                                               3337
      if(flmin.ge.1.0)goto 19821                                           3338
      me=0                                                                 3338
20210 do 20211 j=1,nin                                                     3338
      if(ao(j,1,m).ne.0.0) me=me+1                                         3338
20211 continue                                                             3338
20212 continue                                                             3338
      if(me.gt.ne)goto 19822                                               3339
      if(rsq0-rsq.lt.sml*rsq)goto 19822                                    3339
      if(rsqo(m).gt.rsqmax)goto 19822                                      3340
19821 continue                                                             3341
19822 continue                                                             3341
      deallocate(a,mm,g,ix,del,gj,gk)                                      3342
      return                                                               3343
      end                                                                  3344
      subroutine chkbnds(nr,gk,gkn,xv,cl,al1,al2,a,isc,jerr)               3345
      real gk(nr),cl(2,nr),a(nr)                                           3345
      integer isc(nr)                                                      3346
      kerr=0                                                               3346
      al1p=1.0+al1/xv                                                      3346
      al2p=al2/xv                                                          3346
      isc=0                                                                3347
      gsq=gkn**2                                                           3347
      asq=dot_product(a,a)                                                 3347
      usq=0.0                                                              3348
20220 continue                                                             3348
20221 continue                                                             3348
      vmx=0.0                                                              3349
20230 do 20231 k=1,nr                                                      3349
      v=max(a(k)-cl(2,k),cl(1,k)-a(k))                                     3350
      if(v .le. vmx)goto 20251                                             3350
      vmx=v                                                                3350
      kn=k                                                                 3350
20251 continue                                                             3351
20231 continue                                                             3352
20232 continue                                                             3352
      if(vmx.le.0.0)goto 20222                                             3352
      if(isc(kn).ne.0)goto 20222                                           3353
      gsq=gsq-gk(kn)**2                                                    3353
      g=sqrt(gsq)/xv                                                       3354
      if(a(kn).lt.cl(1,kn)) u=cl(1,kn)                                     3354
      if(a(kn).gt.cl(2,kn)) u=cl(2,kn)                                     3355
      usq=usq+u**2                                                         3356
      if(usq .ne. 0.0)goto 20271                                           3356
      b=max(0.0,(g-al2p)/al1p)                                             3356
      goto 20281                                                           3357
20271 continue                                                             3357
      b0=sqrt(asq-a(kn)**2)                                                3358
      b=bnorm(b0,al1p,al2p,g,usq,kerr)                                     3358
      if(kerr.ne.0)goto 20222                                              3359
20281 continue                                                             3360
20261 continue                                                             3360
      asq=usq+b**2                                                         3360
      if(asq .gt. 0.0)goto 20301                                           3360
      a=0.0                                                                3360
      goto 20222                                                           3360
20301 continue                                                             3361
      a(kn)=u                                                              3361
      isc(kn)=1                                                            3361
      f=1.0/(xv*(al1p+al2p/sqrt(asq)))                                     3362
20310 do 20311 j=1,nr                                                      3362
      if(isc(j).eq.0) a(j)=f*gk(j)                                         3362
20311 continue                                                             3363
20312 continue                                                             3363
      goto 20221                                                           3364
20222 continue                                                             3364
      if(kerr.ne.0) jerr=kerr                                              3365
      return                                                               3366
      end                                                                  3367
      subroutine chkbnds1(nr,gk,gkn,xv,cl1,cl2,al1,al2,a,isc,jerr)         3368
      real gk(nr),a(nr)                                                    3368
      integer isc(nr)                                                      3369
      kerr=0                                                               3369
      al1p=1.0+al1/xv                                                      3369
      al2p=al2/xv                                                          3369
      isc=0                                                                3370
      gsq=gkn**2                                                           3370
      asq=dot_product(a,a)                                                 3370
      usq=0.0                                                              3371
20320 continue                                                             3371
20321 continue                                                             3371
      vmx=0.0                                                              3372
20330 do 20331 k=1,nr                                                      3372
      v=max(a(k)-cl2,cl1-a(k))                                             3373
      if(v .le. vmx)goto 20351                                             3373
      vmx=v                                                                3373
      kn=k                                                                 3373
20351 continue                                                             3374
20331 continue                                                             3375
20332 continue                                                             3375
      if(vmx.le.0.0)goto 20322                                             3375
      if(isc(kn).ne.0)goto 20322                                           3376
      gsq=gsq-gk(kn)**2                                                    3376
      g=sqrt(gsq)/xv                                                       3377
      if(a(kn).lt.cl1) u=cl1                                               3377
      if(a(kn).gt.cl2) u=cl2                                               3378
      usq=usq+u**2                                                         3379
      if(usq .ne. 0.0)goto 20371                                           3379
      b=max(0.0,(g-al2p)/al1p)                                             3379
      goto 20381                                                           3380
20371 continue                                                             3380
      b0=sqrt(asq-a(kn)**2)                                                3381
      b=bnorm(b0,al1p,al2p,g,usq,kerr)                                     3381
      if(kerr.ne.0)goto 20322                                              3382
20381 continue                                                             3383
20361 continue                                                             3383
      asq=usq+b**2                                                         3383
      if(asq .gt. 0.0)goto 20401                                           3383
      a=0.0                                                                3383
      goto 20322                                                           3383
20401 continue                                                             3384
      a(kn)=u                                                              3384
      isc(kn)=1                                                            3384
      f=1.0/(xv*(al1p+al2p/sqrt(asq)))                                     3385
20410 do 20411 j=1,nr                                                      3385
      if(isc(j).eq.0) a(j)=f*gk(j)                                         3385
20411 continue                                                             3386
20412 continue                                                             3386
      goto 20321                                                           3387
20322 continue                                                             3387
      if(kerr.ne.0) jerr=kerr                                              3388
      return                                                               3389
      end                                                                  3390
      function bnorm(b0,al1p,al2p,g,usq,jerr)                              3391
      data thr,mxit /1.0e-10,100/                                          3392
      b=b0                                                                 3392
      zsq=b**2+usq                                                         3392
      if(zsq .gt. 0.0)goto 20431                                           3392
      bnorm=0.0                                                            3392
      return                                                               3392
20431 continue                                                             3393
      z=sqrt(zsq)                                                          3393
      f=b*(al1p+al2p/z)-g                                                  3393
      jerr=0                                                               3394
20440 do 20441 it=1,mxit                                                   3394
      b=b-f/(al1p+al2p*usq/(z*zsq))                                        3395
      zsq=b**2+usq                                                         3395
      if(zsq .gt. 0.0)goto 20461                                           3395
      bnorm=0.0                                                            3395
      return                                                               3395
20461 continue                                                             3396
      z=sqrt(zsq)                                                          3396
      f=b*(al1p+al2p/z)-g                                                  3397
      if(abs(f).le.thr)goto 20442                                          3397
      if(b .gt. 0.0)goto 20481                                             3397
      b=0.0                                                                3397
      goto 20442                                                           3397
20481 continue                                                             3398
20441 continue                                                             3399
20442 continue                                                             3399
      bnorm=b                                                              3399
      if(it.ge.mxit) jerr=90000                                            3400
      return                                                               3401
      entry chg_bnorm(arg,irg)                                             3401
      chg_bnorm=0.0                                                        3401
      thr=arg                                                              3401
      mxit=irg                                                             3401
      return                                                               3402
      entry get_bnorm(arg,irg)                                             3402
      bnorm=0.0                                                            3401
      arg=thr                                                              3402
      irg=mxit                                                             3402
      return                                                               3403
      end                                                                  3404
      subroutine multsolns(ni,nx,nr,lmu,a,ia,nin,b)                        3405
      real a(nx,nr,lmu),b(ni,nr,lmu)                                       3405
      integer ia(nx),nin(lmu)                                              3406
20490 do 20491 lam=1,lmu                                                   3406
      call multuncomp(ni,nr,nx,a(1,1,lam),ia,nin(lam),b(1,1,lam))          3406
20491 continue                                                             3407
20492 continue                                                             3407
      return                                                               3408
      end                                                                  3409
      subroutine multuncomp(ni,nr,nx,ca,ia,nin,a)                          3410
      real ca(nx,nr),a(ni,nr)                                              3410
      integer ia(nx)                                                       3411
      a=0.0                                                                3412
      if(nin .le. 0)goto 20511                                             3412
20520 do 20521 j=1,nr                                                      3412
      a(ia(1:nin),j)=ca(1:nin,j)                                           3412
20521 continue                                                             3412
20522 continue                                                             3412
20511 continue                                                             3413
      return                                                               3414
      end                                                                  3415
      subroutine multmodval(nx,nr,a0,ca,ia,nin,n,x,f)                      3416
      real a0(nr),ca(nx,nr),x(n,*),f(nr,n)                                 3416
      integer ia(nx)                                                       3417
20530 do 20531 i=1,n                                                       3417
      f(:,i)=a0                                                            3417
20531 continue                                                             3417
20532 continue                                                             3417
      if(nin.le.0) return                                                  3418
20540 do 20541 i=1,n                                                       3418
20550 do 20551 j=1,nr                                                      3418
      f(j,i)=f(j,i)+dot_product(ca(1:nin,j),x(i,ia(1:nin)))                3418
20551 continue                                                             3418
20552 continue                                                             3418
20541 continue                                                             3419
20542 continue                                                             3419
      return                                                               3420
      end                                                                  3421
      subroutine multspelnet  (parm,no,ni,nr,x,ix,jx,y,w,jd,vp,cl,ne,nx,   3424 
     *nlam,flmin,ulam,thr,isd,  jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,
     *nlp,jerr)
      real x(*),y(no,nr),w(no),vp(ni),ulam(nlam),cl(2,ni)                  3425
      real ca(nx,nr,nlam),a0(nr,nlam),rsq(nlam),alm(nlam)                  3426
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           3427
      real, dimension (:), allocatable :: vq;                                   
      if(maxval(vp) .gt. 0.0)goto 20571                                    3430
      jerr=10000                                                           3430
      return                                                               3430
20571 continue                                                             3431
      allocate(vq(1:ni),stat=jerr)                                         3431
      if(jerr.ne.0) return                                                 3432
      vq=max(0.0,vp)                                                       3432
      vq=vq*ni/sum(vq)                                                     3433
      call multspelnetn(parm,no,ni,nr,x,ix,jx,y,w,jd,vq,cl,ne,nx,nlam,fl   3435 
     *min,  ulam,thr,isd,jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,nlp,jer
     *r)
      deallocate(vq)                                                       3436
      return                                                               3437
      end                                                                  3438
      subroutine multspelnetn(parm,no,ni,nr,x,ix,jx,y,w,jd,vp,cl,ne,nx,n   3440 
     *lam,flmin,  ulam,thr,isd,jsd,intr,maxit,lmu,a0,ca,ia,nin,rsq,alm,n
     *lp,jerr)
      real x(*),vp(ni),y(no,nr),w(no),ulam(nlam),cl(2,ni)                  3441
      real ca(nx,nr,nlam),a0(nr,nlam),rsq(nlam),alm(nlam)                  3442
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)                           3443
      real, dimension (:), allocatable :: xm,xs,xv,ym,ys                        
      integer, dimension (:), allocatable :: ju                                 
      real, dimension (:,:,:), allocatable :: clt                               
      allocate(clt(1:2,1:nr,1:ni),stat=jerr)                                    
      allocate(xm(1:ni),stat=ierr)                                         3449
      jerr=jerr+ierr                                                       3450
      allocate(xs(1:ni),stat=ierr)                                         3450
      jerr=jerr+ierr                                                       3451
      allocate(ym(1:nr),stat=ierr)                                         3451
      jerr=jerr+ierr                                                       3452
      allocate(ys(1:nr),stat=ierr)                                         3452
      jerr=jerr+ierr                                                       3453
      allocate(ju(1:ni),stat=ierr)                                         3453
      jerr=jerr+ierr                                                       3454
      allocate(xv(1:ni),stat=ierr)                                         3454
      jerr=jerr+ierr                                                       3455
      if(jerr.ne.0) return                                                 3456
      call spchkvars(no,ni,x,ix,ju)                                        3457
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0                                 3458
      if(maxval(ju) .gt. 0)goto 20591                                      3458
      jerr=7777                                                            3458
      return                                                               3458
20591 continue                                                             3459
      call multspstandard1(no,ni,nr,x,ix,jx,y,w,ju,isd,jsd,intr,  xm,xs,   3461 
     *ym,ys,xv,ys0,jerr)
      if(jerr.ne.0) return                                                 3462
20600 do 20601 j=1,ni                                                      3462
20610 do 20611 k=1,nr                                                      3462
20620 do 20621 i=1,2                                                       3462
      clt(i,k,j)=cl(i,j)                                                   3462
20621 continue                                                             3462
20622 continue                                                             3462
20611 continue                                                             3462
20612 continue                                                             3462
20601 continue                                                             3463
20602 continue                                                             3463
      if(isd .le. 0)goto 20641                                             3463
20650 do 20651 j=1,ni                                                      3463
20660 do 20661 k=1,nr                                                      3463
20670 do 20671 i=1,2                                                       3463
      clt(i,k,j)=clt(i,k,j)*xs(j)                                          3463
20671 continue                                                             3463
20672 continue                                                             3463
20661 continue                                                             3463
20662 continue                                                             3463
20651 continue                                                             3463
20652 continue                                                             3463
20641 continue                                                             3464
      if(jsd .le. 0)goto 20691                                             3464
20700 do 20701 j=1,ni                                                      3464
20710 do 20711 k=1,nr                                                      3464
20720 do 20721 i=1,2                                                       3464
      clt(i,k,j)=clt(i,k,j)/ys(k)                                          3464
20721 continue                                                             3464
20722 continue                                                             3464
20711 continue                                                             3464
20712 continue                                                             3464
20701 continue                                                             3464
20702 continue                                                             3464
20691 continue                                                             3465
      call multspelnet2(parm,ni,nr,y,w,no,ne,nx,x,ix,jx,ju,vp,clt,nlam,f   3467 
     *lmin,  ulam,thr,maxit,xm,xs,xv,ys0,lmu,ca,ia,nin,rsq,alm,nlp,jerr)
      if(jerr.gt.0) return                                                 3468
20730 do 20731 k=1,lmu                                                     3468
      nk=nin(k)                                                            3469
20740 do 20741 j=1,nr                                                      3470
20750 do 20751 l=1,nk                                                      3470
      ca(l,j,k)=ys(j)*ca(l,j,k)/xs(ia(l))                                  3470
20751 continue                                                             3471
20752 continue                                                             3471
      if(intr .ne. 0)goto 20771                                            3471
      a0(j,k)=0.0                                                          3471
      goto 20781                                                           3472
20771 continue                                                             3472
      a0(j,k)=ym(j)-dot_product(ca(1:nk,j,k),xm(ia(1:nk)))                 3472
20781 continue                                                             3473
20761 continue                                                             3473
20741 continue                                                             3474
20742 continue                                                             3474
20731 continue                                                             3475
20732 continue                                                             3475
      deallocate(xm,xs,ym,ys,ju,xv,clt)                                    3476
      return                                                               3477
      end                                                                  3478
      subroutine multspstandard1(no,ni,nr,x,ix,jx,y,w,ju,isd,jsd,intr,     3480 
     *xm,xs,ym,ys,xv,ys0,jerr)
      real x(*),y(no,nr),w(no),xm(ni),xs(ni),xv(ni),ym(nr),ys(nr)          3481
      integer ix(*),jx(*),ju(ni)                                           3482
      w=w/sum(w)                                                           3483
      if(intr .ne. 0)goto 20801                                            3484
20810 do 20811 j=1,ni                                                      3484
      if(ju(j).eq.0)goto 20811                                             3484
      xm(j)=0.0                                                            3484
      jb=ix(j)                                                             3484
      je=ix(j+1)-1                                                         3485
      z=dot_product(w(jx(jb:je)),x(jb:je)**2)                              3486
      if(isd .le. 0)goto 20831                                             3486
      xbq=dot_product(w(jx(jb:je)),x(jb:je))**2                            3486
      vc=z-xbq                                                             3487
      xs(j)=sqrt(vc)                                                       3487
      xv(j)=1.0+xbq/vc                                                     3488
      goto 20841                                                           3489
20831 continue                                                             3489
      xs(j)=1.0                                                            3489
      xv(j)=z                                                              3489
20841 continue                                                             3490
20821 continue                                                             3490
20811 continue                                                             3491
20812 continue                                                             3491
      ys0=0.0                                                              3492
20850 do 20851 j=1,nr                                                      3492
      ym(j)=0.0                                                            3492
      z=dot_product(w,y(:,j)**2)                                           3493
      if(jsd .le. 0)goto 20871                                             3493
      u=z-dot_product(w,y(:,j))**2                                         3493
      ys0=ys0+z/u                                                          3494
      ys(j)=sqrt(u)                                                        3494
      y(:,j)=y(:,j)/ys(j)                                                  3495
      goto 20881                                                           3496
20871 continue                                                             3496
      ys(j)=1.0                                                            3496
      ys0=ys0+z                                                            3496
20881 continue                                                             3497
20861 continue                                                             3497
20851 continue                                                             3498
20852 continue                                                             3498
      return                                                               3499
20801 continue                                                             3500
20890 do 20891 j=1,ni                                                      3500
      if(ju(j).eq.0)goto 20891                                             3501
      jb=ix(j)                                                             3501
      je=ix(j+1)-1                                                         3501
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))                             3502
      xv(j)=dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j)**2                 3503
      if(isd.gt.0) xs(j)=sqrt(xv(j))                                       3504
20891 continue                                                             3505
20892 continue                                                             3505
      if(isd .ne. 0)goto 20911                                             3505
      xs=1.0                                                               3505
      goto 20921                                                           3505
20911 continue                                                             3505
      xv=1.0                                                               3505
20921 continue                                                             3506
20901 continue                                                             3506
      ys0=0.0                                                              3507
20930 do 20931 j=1,nr                                                      3508
      ym(j)=dot_product(w,y(:,j))                                          3508
      y(:,j)=y(:,j)-ym(j)                                                  3509
      z=dot_product(w,y(:,j)**2)                                           3510
      if(jsd .le. 0)goto 20951                                             3510
      ys(j)=sqrt(z)                                                        3510
      y(:,j)=y(:,j)/ys(j)                                                  3510
      goto 20961                                                           3511
20951 continue                                                             3511
      ys0=ys0+z                                                            3511
20961 continue                                                             3512
20941 continue                                                             3512
20931 continue                                                             3513
20932 continue                                                             3513
      if(jsd .ne. 0)goto 20981                                             3513
      ys=1.0                                                               3513
      goto 20991                                                           3513
20981 continue                                                             3513
      ys0=nr                                                               3513
20991 continue                                                             3514
20971 continue                                                             3514
      return                                                               3515
      end                                                                  3516
      subroutine multspelnet2(beta,ni,nr,y,w,no,ne,nx,x,ix,jx,ju,vp,cl,n   3518 
     *lam,flmin,  ulam,thri,maxit,xm,xs,xv,ys0,lmu,ao,ia,kin,rsqo,almo,n
     *lp,jerr)
      real y(no,nr),w(no),x(*),vp(ni),ulam(nlam),cl(2,nr,ni)               3519
      real ao(nx,nr,nlam),rsqo(nlam),almo(nlam),xm(ni),xs(ni),xv(ni)       3520
      integer ix(*),jx(*),ju(ni),ia(nx),kin(nlam)                          3521
      real, dimension (:), allocatable :: g,gj,gk,del,o                         
      integer, dimension (:), allocatable :: mm,iy,isc                          
      real, dimension (:,:), allocatable :: a                                   
      allocate(a(1:nr,1:ni),stat=jerr)                                          
      call get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)               3528
      allocate(mm(1:ni),stat=ierr)                                         3528
      jerr=jerr+ierr                                                       3529
      allocate(g(1:ni),stat=ierr)                                          3529
      jerr=jerr+ierr                                                       3530
      allocate(gj(1:nr),stat=ierr)                                         3530
      jerr=jerr+ierr                                                       3531
      allocate(gk(1:nr),stat=ierr)                                         3531
      jerr=jerr+ierr                                                       3532
      allocate(del(1:nr),stat=ierr)                                        3532
      jerr=jerr+ierr                                                       3533
      allocate(o(1:nr),stat=ierr)                                          3533
      jerr=jerr+ierr                                                       3534
      allocate(iy(1:ni),stat=ierr)                                         3534
      jerr=jerr+ierr                                                       3535
      allocate(isc(1:nr),stat=ierr)                                        3535
      jerr=jerr+ierr                                                       3536
      if(jerr.ne.0) return                                                 3537
      bta=beta                                                             3537
      omb=1.0-bta                                                          3537
      alm=0.0                                                              3537
      iy=0                                                                 3537
      thr=thri*ys0/nr                                                      3538
      if(flmin .ge. 1.0)goto 21011                                         3538
      eqs=max(eps,flmin)                                                   3538
      alf=eqs**(1.0/(nlam-1))                                              3538
21011 continue                                                             3539
      rsq=ys0                                                              3539
      a=0.0                                                                3539
      mm=0                                                                 3539
      o=0.0                                                                3539
      nlp=0                                                                3539
      nin=nlp                                                              3539
      iz=0                                                                 3539
      mnl=min(mnlam,nlam)                                                  3540
21020 do 21021 j=1,ni                                                      3540
      if(ju(j).eq.0)goto 21021                                             3540
      jb=ix(j)                                                             3540
      je=ix(j+1)-1                                                         3540
      g(j)=0.0                                                             3541
21030 do 21031 k=1,nr                                                      3542
      g(j)=g(j)+(dot_product(y(jx(jb:je),k),w(jx(jb:je))*x(jb:je))/xs(j)   3543 
     *)**2
21031 continue                                                             3544
21032 continue                                                             3544
      g(j)=sqrt(g(j))                                                      3545
21021 continue                                                             3546
21022 continue                                                             3546
21040 do 21041 m=1,nlam                                                    3546
      alm0=alm                                                             3547
      if(flmin .lt. 1.0)goto 21061                                         3547
      alm=ulam(m)                                                          3547
      goto 21051                                                           3548
21061 if(m .le. 2)goto 21071                                               3548
      alm=alm*alf                                                          3548
      goto 21051                                                           3549
21071 if(m .ne. 1)goto 21081                                               3549
      alm=big                                                              3549
      goto 21091                                                           3550
21081 continue                                                             3550
      alm0=0.0                                                             3551
21100 do 21101 j=1,ni                                                      3551
      if(ju(j).eq.0)goto 21101                                             3552
      if(vp(j).gt.0.0) alm0=max(alm0,g(j)/vp(j))                           3553
21101 continue                                                             3554
21102 continue                                                             3554
      alm0=alm0/max(bta,1.0e-3)                                            3554
      alm=alf*alm0                                                         3555
21091 continue                                                             3556
21051 continue                                                             3556
      dem=alm*omb                                                          3556
      ab=alm*bta                                                           3556
      rsq0=rsq                                                             3556
      jz=1                                                                 3557
      tlam=bta*(2.0*alm-alm0)                                              3558
21110 do 21111 k=1,ni                                                      3558
      if(iy(k).eq.1)goto 21111                                             3558
      if(ju(k).eq.0)goto 21111                                             3559
      if(g(k).gt.tlam*vp(k)) iy(k)=1                                       3560
21111 continue                                                             3561
21112 continue                                                             3561
21120 continue                                                             3561
21121 continue                                                             3561
      if(iz*jz.ne.0) go to 10360                                           3562
10880 continue                                                             3562
      nlp=nlp+1                                                            3562
      dlx=0.0                                                              3563
21130 do 21131 k=1,ni                                                      3563
      if(iy(k).eq.0)goto 21131                                             3563
      jb=ix(k)                                                             3563
      je=ix(k+1)-1                                                         3563
      gkn=0.0                                                              3564
21140 do 21141 j=1,nr                                                      3565
      gj(j)=dot_product(y(jx(jb:je),j)+o(j),w(jx(jb:je))*x(jb:je))/xs(k)   3566
      gk(j)=gj(j)+a(j,k)*xv(k)                                             3566
      gkn=gkn+gk(j)**2                                                     3567
21141 continue                                                             3568
21142 continue                                                             3568
      gkn=sqrt(gkn)                                                        3568
      u=1.0-ab*vp(k)/gkn                                                   3568
      del=a(:,k)                                                           3569
      if(u .gt. 0.0)goto 21161                                             3569
      a(:,k)=0.0                                                           3569
      goto 21171                                                           3570
21161 continue                                                             3570
      a(:,k)=gk*(u/(xv(k)+dem*vp(k)))                                      3571
      call chkbnds(nr,gk,gkn,xv(k),cl(1,1,k),  dem*vp(k),ab*vp(k),a(:,k)   3573 
     *,isc,jerr)
      if(jerr.ne.0) return                                                 3574
21171 continue                                                             3575
21151 continue                                                             3575
      del=a(:,k)-del                                                       3575
      if(maxval(abs(del)).le.0.0)goto 21131                                3576
      if(mm(k) .ne. 0)goto 21191                                           3576
      nin=nin+1                                                            3576
      if(nin.gt.nx)goto 21132                                              3577
      mm(k)=nin                                                            3577
      ia(nin)=k                                                            3578
21191 continue                                                             3579
21200 do 21201 j=1,nr                                                      3579
      rsq=rsq-del(j)*(2.0*gj(j)-del(j)*xv(k))                              3580
      y(jx(jb:je),j)=y(jx(jb:je),j)-del(j)*x(jb:je)/xs(k)                  3581
      o(j)=o(j)+del(j)*xm(k)/xs(k)                                         3581
      dlx=max(xv(k)*del(j)**2,dlx)                                         3582
21201 continue                                                             3583
21202 continue                                                             3583
21131 continue                                                             3584
21132 continue                                                             3584
      if(nin.gt.nx)goto 21122                                              3585
      if(dlx .ge. thr)goto 21221                                           3585
      ixx=0                                                                3586
21230 do 21231 j=1,ni                                                      3586
      if(iy(j).eq.1)goto 21231                                             3586
      if(ju(j).eq.0)goto 21231                                             3587
      jb=ix(j)                                                             3587
      je=ix(j+1)-1                                                         3587
      g(j)=0.0                                                             3588
21240 do 21241 k=1,nr                                                      3588
      g(j)=g(j)+  (dot_product(y(jx(jb:je),k)+o(k),w(jx(jb:je))*x(jb:je)   3590 
     *)/xs(j))**2
21241 continue                                                             3591
21242 continue                                                             3591
      g(j)=sqrt(g(j))                                                      3592
      if(g(j) .le. ab*vp(j))goto 21261                                     3592
      iy(j)=1                                                              3592
      ixx=1                                                                3592
21261 continue                                                             3593
21231 continue                                                             3594
21232 continue                                                             3594
      if(ixx.eq.1) go to 10880                                             3595
      goto 21122                                                           3596
21221 continue                                                             3597
      if(nlp .le. maxit)goto 21281                                         3597
      jerr=-m                                                              3597
      return                                                               3597
21281 continue                                                             3598
10360 continue                                                             3598
      iz=1                                                                 3599
21290 continue                                                             3599
21291 continue                                                             3599
      nlp=nlp+1                                                            3599
      dlx=0.0                                                              3600
21300 do 21301 l=1,nin                                                     3600
      k=ia(l)                                                              3600
      jb=ix(k)                                                             3600
      je=ix(k+1)-1                                                         3600
      gkn=0.0                                                              3601
21310 do 21311 j=1,nr                                                      3601
      gj(j)=  dot_product(y(jx(jb:je),j)+o(j),w(jx(jb:je))*x(jb:je))/xs(   3603 
     *k)
      gk(j)=gj(j)+a(j,k)*xv(k)                                             3603
      gkn=gkn+gk(j)**2                                                     3604
21311 continue                                                             3605
21312 continue                                                             3605
      gkn=sqrt(gkn)                                                        3605
      u=1.0-ab*vp(k)/gkn                                                   3605
      del=a(:,k)                                                           3606
      if(u .gt. 0.0)goto 21331                                             3606
      a(:,k)=0.0                                                           3606
      goto 21341                                                           3607
21331 continue                                                             3607
      a(:,k)=gk*(u/(xv(k)+dem*vp(k)))                                      3608
      call chkbnds(nr,gk,gkn,xv(k),cl(1,1,k),  dem*vp(k),ab*vp(k),a(:,k)   3610 
     *,isc,jerr)
      if(jerr.ne.0) return                                                 3611
21341 continue                                                             3612
21321 continue                                                             3612
      del=a(:,k)-del                                                       3612
      if(maxval(abs(del)).le.0.0)goto 21301                                3613
21350 do 21351 j=1,nr                                                      3613
      rsq=rsq-del(j)*(2.0*gj(j)-del(j)*xv(k))                              3614
      y(jx(jb:je),j)=y(jx(jb:je),j)-del(j)*x(jb:je)/xs(k)                  3615
      o(j)=o(j)+del(j)*xm(k)/xs(k)                                         3615
      dlx=max(xv(k)*del(j)**2,dlx)                                         3616
21351 continue                                                             3617
21352 continue                                                             3617
21301 continue                                                             3618
21302 continue                                                             3618
      if(dlx.lt.thr)goto 21292                                             3618
      if(nlp .le. maxit)goto 21371                                         3618
      jerr=-m                                                              3618
      return                                                               3618
21371 continue                                                             3619
      goto 21291                                                           3620
21292 continue                                                             3620
      jz=0                                                                 3621
      goto 21121                                                           3622
21122 continue                                                             3622
      if(nin .le. nx)goto 21391                                            3622
      jerr=-10000-m                                                        3622
      goto 21042                                                           3622
21391 continue                                                             3623
      if(nin .le. 0)goto 21411                                             3623
21420 do 21421 j=1,nr                                                      3623
      ao(1:nin,j,m)=a(j,ia(1:nin))                                         3623
21421 continue                                                             3623
21422 continue                                                             3623
21411 continue                                                             3624
      kin(m)=nin                                                           3625
      rsqo(m)=1.0-rsq/ys0                                                  3625
      almo(m)=alm                                                          3625
      lmu=m                                                                3626
      if(m.lt.mnl)goto 21041                                               3626
      if(flmin.ge.1.0)goto 21041                                           3627
      me=0                                                                 3627
21430 do 21431 j=1,nin                                                     3627
      if(ao(j,1,m).ne.0.0) me=me+1                                         3627
21431 continue                                                             3627
21432 continue                                                             3627
      if(me.gt.ne)goto 21042                                               3628
      if(rsq0-rsq.lt.sml*rsq)goto 21042                                    3628
      if(rsqo(m).gt.rsqmax)goto 21042                                      3629
21041 continue                                                             3630
21042 continue                                                             3630
      deallocate(a,mm,g,iy,gj,gk,del,o)                                    3631
      return                                                               3632
      end                                                                  3633
      subroutine multlognetn(parm,no,ni,nc,x,y,g,w,ju,vp,cl,ne,nx,nlam,f   3635 
     *lmin,ulam,  shri,intr,maxit,xv,lmu,a0,a,m,kin,dev0,dev,alm,nlp,jer
     *r)
      real x(no,ni),y(no,nc),g(no,nc),w(no),vp(ni),ulam(nlam),cl(2,ni)     3636
      real a(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),xv(ni)            3637
      integer ju(ni),m(nx),kin(nlam)                                       3638
      real, dimension (:,:), allocatable :: q,r,b,bs                            
      real, dimension (:), allocatable :: sxp,sxpl,ga,gk,del                    
      integer, dimension (:), allocatable :: mm,is,ixx,isc                      
      allocate(b(0:ni,1:nc),stat=jerr)                                          
      allocate(bs(0:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(q(1:no,1:nc),stat=ierr); jerr=jerr+ierr                          
      allocate(r(1:no,1:nc),stat=ierr); jerr=jerr+ierr;                         
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               3647
      exmn=-exmx                                                           3648
      allocate(mm(1:ni),stat=ierr)                                         3648
      jerr=jerr+ierr                                                       3649
      allocate(is(1:max(nc,ni)),stat=ierr)                                 3649
      jerr=jerr+ierr                                                       3650
      allocate(sxp(1:no),stat=ierr)                                        3650
      jerr=jerr+ierr                                                       3651
      allocate(sxpl(1:no),stat=ierr)                                       3651
      jerr=jerr+ierr                                                       3652
      allocate(ga(1:ni),stat=ierr)                                         3652
      jerr=jerr+ierr                                                       3653
      allocate(ixx(1:ni),stat=ierr)                                        3653
      jerr=jerr+ierr                                                       3654
      allocate(gk(1:nc),stat=ierr)                                         3654
      jerr=jerr+ierr                                                       3655
      allocate(del(1:nc),stat=ierr)                                        3655
      jerr=jerr+ierr                                                       3656
      allocate(isc(1:nc),stat=ierr)                                        3656
      jerr=jerr+ierr                                                       3657
      if(jerr.ne.0) return                                                 3658
      pmax=1.0-pmin                                                        3658
      emin=pmin/pmax                                                       3658
      emax=1.0/emin                                                        3659
      bta=parm                                                             3659
      omb=1.0-bta                                                          3659
      dev1=0.0                                                             3659
      dev0=0.0                                                             3660
21440 do 21441 ic=1,nc                                                     3660
      q0=dot_product(w,y(:,ic))                                            3661
      if(q0 .gt. pmin)goto 21461                                           3661
      jerr =8000+ic                                                        3661
      return                                                               3661
21461 continue                                                             3662
      if(q0 .lt. pmax)goto 21481                                           3662
      jerr =9000+ic                                                        3662
      return                                                               3662
21481 continue                                                             3663
      if(intr .ne. 0)goto 21501                                            3663
      q0=1.0/nc                                                            3663
      b(0,ic)=0.0                                                          3663
      goto 21511                                                           3664
21501 continue                                                             3664
      b(0,ic)=log(q0)                                                      3664
      dev1=dev1-q0*b(0,ic)                                                 3664
21511 continue                                                             3665
21491 continue                                                             3665
      b(1:ni,ic)=0.0                                                       3666
21441 continue                                                             3667
21442 continue                                                             3667
      if(intr.eq.0) dev1=log(float(nc))                                    3667
      ixx=0                                                                3667
      al=0.0                                                               3668
      if(nonzero(no*nc,g) .ne. 0)goto 21531                                3669
      b(0,:)=b(0,:)-sum(b(0,:))/nc                                         3669
      sxp=0.0                                                              3670
21540 do 21541 ic=1,nc                                                     3670
      q(:,ic)=exp(b(0,ic))                                                 3670
      sxp=sxp+q(:,ic)                                                      3670
21541 continue                                                             3671
21542 continue                                                             3671
      goto 21551                                                           3672
21531 continue                                                             3672
21560 do 21561 i=1,no                                                      3672
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         3672
21561 continue                                                             3672
21562 continue                                                             3672
      sxp=0.0                                                              3673
      if(intr .ne. 0)goto 21581                                            3673
      b(0,:)=0.0                                                           3673
      goto 21591                                                           3674
21581 continue                                                             3674
      call kazero(nc,no,y,g,w,b(0,:),jerr)                                 3674
      if(jerr.ne.0) return                                                 3674
21591 continue                                                             3675
21571 continue                                                             3675
      dev1=0.0                                                             3676
21600 do 21601 ic=1,nc                                                     3676
      q(:,ic)=b(0,ic)+g(:,ic)                                              3677
      dev1=dev1-dot_product(w,y(:,ic)*q(:,ic))                             3678
      q(:,ic)=exp(q(:,ic))                                                 3678
      sxp=sxp+q(:,ic)                                                      3679
21601 continue                                                             3680
21602 continue                                                             3680
      sxpl=w*log(sxp)                                                      3680
21610 do 21611 ic=1,nc                                                     3680
      dev1=dev1+dot_product(y(:,ic),sxpl)                                  3680
21611 continue                                                             3681
21612 continue                                                             3681
21551 continue                                                             3682
21521 continue                                                             3682
21620 do 21621 ic=1,nc                                                     3682
21630 do 21631 i=1,no                                                      3682
      if(y(i,ic).gt.0.0) dev0=dev0+w(i)*y(i,ic)*log(y(i,ic))               3682
21631 continue                                                             3682
21632 continue                                                             3682
21621 continue                                                             3683
21622 continue                                                             3683
      dev0=dev0+dev1                                                       3684
      if(flmin .ge. 1.0)goto 21651                                         3684
      eqs=max(eps,flmin)                                                   3684
      alf=eqs**(1.0/(nlam-1))                                              3684
21651 continue                                                             3685
      m=0                                                                  3685
      mm=0                                                                 3685
      nin=0                                                                3685
      nlp=0                                                                3685
      mnl=min(mnlam,nlam)                                                  3685
      bs=0.0                                                               3685
      shr=shri*dev0                                                        3686
      ga=0.0                                                               3687
21660 do 21661 ic=1,nc                                                     3687
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)                                      3688
21670 do 21671 j=1,ni                                                      3688
      if(ju(j).ne.0) ga(j)=ga(j)+dot_product(r(:,ic),x(:,j))**2            3688
21671 continue                                                             3689
21672 continue                                                             3689
21661 continue                                                             3690
21662 continue                                                             3690
      ga=sqrt(ga)                                                          3691
21680 do 21681 ilm=1,nlam                                                  3691
      al0=al                                                               3692
      if(flmin .lt. 1.0)goto 21701                                         3692
      al=ulam(ilm)                                                         3692
      goto 21691                                                           3693
21701 if(ilm .le. 2)goto 21711                                             3693
      al=al*alf                                                            3693
      goto 21691                                                           3694
21711 if(ilm .ne. 1)goto 21721                                             3694
      al=big                                                               3694
      goto 21731                                                           3695
21721 continue                                                             3695
      al0=0.0                                                              3696
21740 do 21741 j=1,ni                                                      3696
      if(ju(j).eq.0)goto 21741                                             3696
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            3696
21741 continue                                                             3697
21742 continue                                                             3697
      al0=al0/max(bta,1.0e-3)                                              3697
      al=alf*al0                                                           3698
21731 continue                                                             3699
21691 continue                                                             3699
      al2=al*omb                                                           3699
      al1=al*bta                                                           3699
      tlam=bta*(2.0*al-al0)                                                3700
21750 do 21751 k=1,ni                                                      3700
      if(ixx(k).eq.1)goto 21751                                            3700
      if(ju(k).eq.0)goto 21751                                             3701
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1                                     3702
21751 continue                                                             3703
21752 continue                                                             3703
10880 continue                                                             3704
21760 continue                                                             3704
21761 continue                                                             3704
      ix=0                                                                 3704
      jx=ix                                                                3704
      kx=jx                                                                3704
      t=0.0                                                                3705
21770 do 21771 ic=1,nc                                                     3705
      t=max(t,maxval(q(:,ic)*(1.0-q(:,ic)/sxp)/sxp))                       3705
21771 continue                                                             3706
21772 continue                                                             3706
      if(t .ge. eps)goto 21791                                             3706
      kx=1                                                                 3706
      goto 21762                                                           3706
21791 continue                                                             3706
      t=2.0*t                                                              3706
      alt=al1/t                                                            3706
      al2t=al2/t                                                           3707
21800 do 21801 ic=1,nc                                                     3708
      bs(0,ic)=b(0,ic)                                                     3708
      if(nin.gt.0) bs(m(1:nin),ic)=b(m(1:nin),ic)                          3709
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)/t                                    3710
      d=0.0                                                                3710
      if(intr.ne.0) d=sum(r(:,ic))                                         3711
      if(d .eq. 0.0)goto 21821                                             3712
      b(0,ic)=b(0,ic)+d                                                    3712
      r(:,ic)=r(:,ic)-d*w                                                  3712
      dlx=max(dlx,d**2)                                                    3713
21821 continue                                                             3714
21801 continue                                                             3715
21802 continue                                                             3715
21830 continue                                                             3715
21831 continue                                                             3715
      nlp=nlp+nc                                                           3715
      dlx=0.0                                                              3716
21840 do 21841 k=1,ni                                                      3716
      if(ixx(k).eq.0)goto 21841                                            3716
      gkn=0.0                                                              3717
21850 do 21851 ic=1,nc                                                     3717
      gk(ic)=dot_product(r(:,ic),x(:,k))+b(k,ic)*xv(k)                     3718
      gkn=gkn+gk(ic)**2                                                    3719
21851 continue                                                             3720
21852 continue                                                             3720
      gkn=sqrt(gkn)                                                        3720
      u=1.0-alt*vp(k)/gkn                                                  3720
      del=b(k,:)                                                           3721
      if(u .gt. 0.0)goto 21871                                             3721
      b(k,:)=0.0                                                           3721
      goto 21881                                                           3722
21871 continue                                                             3722
      b(k,:)=gk*(u/(xv(k)+vp(k)*al2t))                                     3723
      call chkbnds1(nc,gk,gkn,xv(k),cl(1,k),  cl(2,k),vp(k)*al2t,alt*vp(   3725 
     *k),b(k,:),isc,jerr)
      if(jerr.ne.0) return                                                 3726
21881 continue                                                             3727
21861 continue                                                             3727
      del=b(k,:)-del                                                       3727
      if(maxval(abs(del)).le.0.0)goto 21841                                3728
21890 do 21891 ic=1,nc                                                     3728
      dlx=max(dlx,xv(k)*del(ic)**2)                                        3729
      r(:,ic)=r(:,ic)-del(ic)*w*x(:,k)                                     3730
21891 continue                                                             3731
21892 continue                                                             3731
      if(mm(k) .ne. 0)goto 21911                                           3731
      nin=nin+1                                                            3732
      if(nin .le. nx)goto 21931                                            3732
      jx=1                                                                 3732
      goto 21842                                                           3732
21931 continue                                                             3733
      mm(k)=nin                                                            3733
      m(nin)=k                                                             3734
21911 continue                                                             3735
21841 continue                                                             3736
21842 continue                                                             3736
      if(jx.gt.0)goto 21832                                                3736
      if(dlx.lt.shr)goto 21832                                             3737
      if(nlp .le. maxit)goto 21951                                         3737
      jerr=-ilm                                                            3737
      return                                                               3737
21951 continue                                                             3738
21960 continue                                                             3738
21961 continue                                                             3738
      nlp=nlp+nc                                                           3738
      dlx=0.0                                                              3739
21970 do 21971 l=1,nin                                                     3739
      k=m(l)                                                               3739
      gkn=0.0                                                              3740
21980 do 21981 ic=1,nc                                                     3740
      gk(ic)=dot_product(r(:,ic),x(:,k))+b(k,ic)*xv(k)                     3741
      gkn=gkn+gk(ic)**2                                                    3742
21981 continue                                                             3743
21982 continue                                                             3743
      gkn=sqrt(gkn)                                                        3743
      u=1.0-alt*vp(k)/gkn                                                  3743
      del=b(k,:)                                                           3744
      if(u .gt. 0.0)goto 22001                                             3744
      b(k,:)=0.0                                                           3744
      goto 22011                                                           3745
22001 continue                                                             3745
      b(k,:)=gk*(u/(xv(k)+vp(k)*al2t))                                     3746
      call chkbnds1(nc,gk,gkn,xv(k),cl(1,k),  cl(2,k),vp(k)*al2t,alt*vp(   3748 
     *k),b(k,:),isc,jerr)
      if(jerr.ne.0) return                                                 3749
22011 continue                                                             3750
21991 continue                                                             3750
      del=b(k,:)-del                                                       3750
      if(maxval(abs(del)).le.0.0)goto 21971                                3751
22020 do 22021 ic=1,nc                                                     3751
      dlx=max(dlx,xv(k)*del(ic)**2)                                        3752
      r(:,ic)=r(:,ic)-del(ic)*w*x(:,k)                                     3753
22021 continue                                                             3754
22022 continue                                                             3754
21971 continue                                                             3755
21972 continue                                                             3755
      if(dlx.lt.shr)goto 21962                                             3755
      if(nlp .le. maxit)goto 22041                                         3755
      jerr=-ilm                                                            3755
      return                                                               3755
22041 continue                                                             3757
      goto 21961                                                           3758
21962 continue                                                             3758
      goto 21831                                                           3759
21832 continue                                                             3759
      if(jx.gt.0)goto 21762                                                3760
22050 do 22051 ic=1,nc                                                     3761
      if((b(0,ic)-bs(0,ic))**2.gt.shr) ix=1                                3762
      if(ix .ne. 0)goto 22071                                              3763
22080 do 22081 j=1,nin                                                     3763
      k=m(j)                                                               3764
      if(xv(k)*(b(k,ic)-bs(k,ic))**2 .le. shr)goto 22101                   3764
      ix=1                                                                 3764
      goto 22082                                                           3764
22101 continue                                                             3766
22081 continue                                                             3767
22082 continue                                                             3767
22071 continue                                                             3768
22110 do 22111 i=1,no                                                      3768
      fi=b(0,ic)+g(i,ic)                                                   3770
      if(nin.gt.0) fi=fi+dot_product(b(m(1:nin),ic),x(i,m(1:nin)))         3771
      fi=min(max(exmn,fi),exmx)                                            3771
      sxp(i)=sxp(i)-q(i,ic)                                                3772
      q(i,ic)=min(max(emin*sxp(i),exp(fi)),emax*sxp(i))                    3773
      sxp(i)=sxp(i)+q(i,ic)                                                3774
22111 continue                                                             3775
22112 continue                                                             3775
22051 continue                                                             3776
22052 continue                                                             3776
      s=-sum(b(0,:))/nc                                                    3776
      b(0,:)=b(0,:)+s                                                      3777
      if(jx.gt.0)goto 21762                                                3778
      if(ix .ne. 0)goto 22131                                              3779
22140 do 22141 k=1,ni                                                      3779
      if(ixx(k).eq.1)goto 22141                                            3779
      if(ju(k).eq.0)goto 22141                                             3779
      ga(k)=0.0                                                            3779
22141 continue                                                             3780
22142 continue                                                             3780
22150 do 22151 ic=1,nc                                                     3780
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)                                      3781
22160 do 22161 k=1,ni                                                      3781
      if(ixx(k).eq.1)goto 22161                                            3781
      if(ju(k).eq.0)goto 22161                                             3782
      ga(k)=ga(k)+dot_product(r(:,ic),x(:,k))**2                           3783
22161 continue                                                             3784
22162 continue                                                             3784
22151 continue                                                             3785
22152 continue                                                             3785
      ga=sqrt(ga)                                                          3786
22170 do 22171 k=1,ni                                                      3786
      if(ixx(k).eq.1)goto 22171                                            3786
      if(ju(k).eq.0)goto 22171                                             3787
      if(ga(k) .le. al1*vp(k))goto 22191                                   3787
      ixx(k)=1                                                             3787
      ix=1                                                                 3787
22191 continue                                                             3788
22171 continue                                                             3789
22172 continue                                                             3789
      if(ix.eq.1) go to 10880                                              3790
      goto 21762                                                           3791
22131 continue                                                             3792
      goto 21761                                                           3793
21762 continue                                                             3793
      if(kx .le. 0)goto 22211                                              3793
      jerr=-20000-ilm                                                      3793
      goto 21682                                                           3793
22211 continue                                                             3794
      if(jx .le. 0)goto 22231                                              3794
      jerr=-10000-ilm                                                      3794
      goto 21682                                                           3794
22231 continue                                                             3794
      devi=0.0                                                             3795
22240 do 22241 ic=1,nc                                                     3796
      if(nin.gt.0) a(1:nin,ic,ilm)=b(m(1:nin),ic)                          3796
      a0(ic,ilm)=b(0,ic)                                                   3797
22250 do 22251 i=1,no                                                      3797
      if(y(i,ic).le.0.0)goto 22251                                         3798
      devi=devi-w(i)*y(i,ic)*log(q(i,ic)/sxp(i))                           3799
22251 continue                                                             3800
22252 continue                                                             3800
22241 continue                                                             3801
22242 continue                                                             3801
      kin(ilm)=nin                                                         3801
      alm(ilm)=al                                                          3801
      lmu=ilm                                                              3802
      dev(ilm)=(dev1-devi)/dev0                                            3803
      if(ilm.lt.mnl)goto 21681                                             3803
      if(flmin.ge.1.0)goto 21681                                           3804
      me=0                                                                 3804
22260 do 22261 j=1,nin                                                     3804
      if(a(j,1,ilm).ne.0.0) me=me+1                                        3804
22261 continue                                                             3804
22262 continue                                                             3804
      if(me.gt.ne)goto 21682                                               3805
      if(dev(ilm).gt.devmax)goto 21682                                     3805
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 21682                             3806
21681 continue                                                             3807
21682 continue                                                             3807
      g=log(q)                                                             3807
22270 do 22271 i=1,no                                                      3807
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         3807
22271 continue                                                             3808
22272 continue                                                             3808
      deallocate(sxp,b,bs,r,q,mm,is,ga,ixx,gk,del,sxpl)                    3809
      return                                                               3810
      end                                                                  3811
      subroutine multsprlognetn(parm,no,ni,nc,x,ix,jx,y,g,w,ju,vp,cl,ne,   3813 
     *nx,nlam,  flmin,ulam,shri,intr,maxit,xv,xb,xs,lmu,a0,a,m,kin,dev0,
     *dev,alm,nlp,jerr)
      real x(*),y(no,nc),g(no,nc),w(no),vp(ni),ulam(nlam),xb(ni),xs(ni),   3814 
     *xv(ni)
      real a(nx,nc,nlam),a0(nc,nlam),dev(nlam),alm(nlam),cl(2,ni)          3815
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)                           3816
      real, dimension (:,:), allocatable :: q,r,b,bs                            
      real, dimension (:), allocatable :: sxp,sxpl,ga,gk,del,sc,svr             
      integer, dimension (:), allocatable :: mm,is,iy,isc                       
      allocate(b(0:ni,1:nc),stat=jerr)                                          
      allocate(bs(0:ni,1:nc),stat=ierr); jerr=jerr+ierr                         
      allocate(q(1:no,1:nc),stat=ierr); jerr=jerr+ierr                          
      allocate(r(1:no,1:nc),stat=ierr); jerr=jerr+ierr                          
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)               3825
      exmn=-exmx                                                           3826
      allocate(mm(1:ni),stat=ierr)                                         3826
      jerr=jerr+ierr                                                       3827
      allocate(ga(1:ni),stat=ierr)                                         3827
      jerr=jerr+ierr                                                       3828
      allocate(gk(1:nc),stat=ierr)                                         3828
      jerr=jerr+ierr                                                       3829
      allocate(del(1:nc),stat=ierr)                                        3829
      jerr=jerr+ierr                                                       3830
      allocate(iy(1:ni),stat=ierr)                                         3830
      jerr=jerr+ierr                                                       3831
      allocate(is(1:max(nc,ni)),stat=ierr)                                 3831
      jerr=jerr+ierr                                                       3832
      allocate(sxp(1:no),stat=ierr)                                        3832
      jerr=jerr+ierr                                                       3833
      allocate(sxpl(1:no),stat=ierr)                                       3833
      jerr=jerr+ierr                                                       3834
      allocate(svr(1:nc),stat=ierr)                                        3834
      jerr=jerr+ierr                                                       3835
      allocate(sc(1:no),stat=ierr)                                         3835
      jerr=jerr+ierr                                                       3836
      allocate(isc(1:nc),stat=ierr)                                        3836
      jerr=jerr+ierr                                                       3837
      if(jerr.ne.0) return                                                 3838
      pmax=1.0-pmin                                                        3838
      emin=pmin/pmax                                                       3838
      emax=1.0/emin                                                        3839
      bta=parm                                                             3839
      omb=1.0-bta                                                          3839
      dev1=0.0                                                             3839
      dev0=0.0                                                             3840
22280 do 22281 ic=1,nc                                                     3840
      q0=dot_product(w,y(:,ic))                                            3841
      if(q0 .gt. pmin)goto 22301                                           3841
      jerr =8000+ic                                                        3841
      return                                                               3841
22301 continue                                                             3842
      if(q0 .lt. pmax)goto 22321                                           3842
      jerr =9000+ic                                                        3842
      return                                                               3842
22321 continue                                                             3843
      b(1:ni,ic)=0.0                                                       3844
      if(intr .ne. 0)goto 22341                                            3844
      q0=1.0/nc                                                            3844
      b(0,ic)=0.0                                                          3844
      goto 22351                                                           3845
22341 continue                                                             3845
      b(0,ic)=log(q0)                                                      3845
      dev1=dev1-q0*b(0,ic)                                                 3845
22351 continue                                                             3846
22331 continue                                                             3846
22281 continue                                                             3847
22282 continue                                                             3847
      if(intr.eq.0) dev1=log(float(nc))                                    3847
      iy=0                                                                 3847
      al=0.0                                                               3848
      if(nonzero(no*nc,g) .ne. 0)goto 22371                                3849
      b(0,:)=b(0,:)-sum(b(0,:))/nc                                         3849
      sxp=0.0                                                              3850
22380 do 22381 ic=1,nc                                                     3850
      q(:,ic)=exp(b(0,ic))                                                 3850
      sxp=sxp+q(:,ic)                                                      3850
22381 continue                                                             3851
22382 continue                                                             3851
      goto 22391                                                           3852
22371 continue                                                             3852
22400 do 22401 i=1,no                                                      3852
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         3852
22401 continue                                                             3852
22402 continue                                                             3852
      sxp=0.0                                                              3853
      if(intr .ne. 0)goto 22421                                            3853
      b(0,:)=0.0                                                           3853
      goto 22431                                                           3854
22421 continue                                                             3854
      call kazero(nc,no,y,g,w,b(0,:),jerr)                                 3854
      if(jerr.ne.0) return                                                 3854
22431 continue                                                             3855
22411 continue                                                             3855
      dev1=0.0                                                             3856
22440 do 22441 ic=1,nc                                                     3856
      q(:,ic)=b(0,ic)+g(:,ic)                                              3857
      dev1=dev1-dot_product(w,y(:,ic)*q(:,ic))                             3858
      q(:,ic)=exp(q(:,ic))                                                 3858
      sxp=sxp+q(:,ic)                                                      3859
22441 continue                                                             3860
22442 continue                                                             3860
      sxpl=w*log(sxp)                                                      3860
22450 do 22451 ic=1,nc                                                     3860
      dev1=dev1+dot_product(y(:,ic),sxpl)                                  3860
22451 continue                                                             3861
22452 continue                                                             3861
22391 continue                                                             3862
22361 continue                                                             3862
22460 do 22461 ic=1,nc                                                     3862
22470 do 22471 i=1,no                                                      3862
      if(y(i,ic).gt.0.0) dev0=dev0+w(i)*y(i,ic)*log(y(i,ic))               3862
22471 continue                                                             3862
22472 continue                                                             3862
22461 continue                                                             3863
22462 continue                                                             3863
      dev0=dev0+dev1                                                       3864
      if(flmin .ge. 1.0)goto 22491                                         3864
      eqs=max(eps,flmin)                                                   3864
      alf=eqs**(1.0/(nlam-1))                                              3864
22491 continue                                                             3865
      m=0                                                                  3865
      mm=0                                                                 3865
      nin=0                                                                3865
      nlp=0                                                                3865
      mnl=min(mnlam,nlam)                                                  3865
      bs=0.0                                                               3866
      shr=shri*dev0                                                        3866
      ga=0.0                                                               3867
22500 do 22501 ic=1,nc                                                     3867
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)                                      3867
      svr(ic)=sum(r(:,ic))                                                 3868
22510 do 22511 j=1,ni                                                      3868
      if(ju(j).eq.0)goto 22511                                             3869
      jb=ix(j)                                                             3869
      je=ix(j+1)-1                                                         3870
      gj=dot_product(r(jx(jb:je),ic),x(jb:je))                             3871
      ga(j)=ga(j)+((gj-svr(ic)*xb(j))/xs(j))**2                            3872
22511 continue                                                             3873
22512 continue                                                             3873
22501 continue                                                             3874
22502 continue                                                             3874
      ga=sqrt(ga)                                                          3875
22520 do 22521 ilm=1,nlam                                                  3875
      al0=al                                                               3876
      if(flmin .lt. 1.0)goto 22541                                         3876
      al=ulam(ilm)                                                         3876
      goto 22531                                                           3877
22541 if(ilm .le. 2)goto 22551                                             3877
      al=al*alf                                                            3877
      goto 22531                                                           3878
22551 if(ilm .ne. 1)goto 22561                                             3878
      al=big                                                               3878
      goto 22571                                                           3879
22561 continue                                                             3879
      al0=0.0                                                              3880
22580 do 22581 j=1,ni                                                      3880
      if(ju(j).eq.0)goto 22581                                             3880
      if(vp(j).gt.0.0) al0=max(al0,ga(j)/vp(j))                            3880
22581 continue                                                             3881
22582 continue                                                             3881
      al0=al0/max(bta,1.0e-3)                                              3881
      al=alf*al0                                                           3882
22571 continue                                                             3883
22531 continue                                                             3883
      al2=al*omb                                                           3883
      al1=al*bta                                                           3883
      tlam=bta*(2.0*al-al0)                                                3884
22590 do 22591 k=1,ni                                                      3884
      if(iy(k).eq.1)goto 22591                                             3884
      if(ju(k).eq.0)goto 22591                                             3885
      if(ga(k).gt.tlam*vp(k)) iy(k)=1                                      3886
22591 continue                                                             3887
22592 continue                                                             3887
10880 continue                                                             3888
22600 continue                                                             3888
22601 continue                                                             3888
      ixx=0                                                                3888
      jxx=ixx                                                              3888
      kxx=jxx                                                              3888
      t=0.0                                                                3889
22610 do 22611 ic=1,nc                                                     3889
      t=max(t,maxval(q(:,ic)*(1.0-q(:,ic)/sxp)/sxp))                       3889
22611 continue                                                             3890
22612 continue                                                             3890
      if(t .ge. eps)goto 22631                                             3890
      kxx=1                                                                3890
      goto 22602                                                           3890
22631 continue                                                             3890
      t=2.0*t                                                              3890
      alt=al1/t                                                            3890
      al2t=al2/t                                                           3891
22640 do 22641 ic=1,nc                                                     3891
      bs(0,ic)=b(0,ic)                                                     3891
      if(nin.gt.0) bs(m(1:nin),ic)=b(m(1:nin),ic)                          3892
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)/t                                    3892
      svr(ic)=sum(r(:,ic))                                                 3893
      if(intr .eq. 0)goto 22661                                            3893
      b(0,ic)=b(0,ic)+svr(ic)                                              3893
      r(:,ic)=r(:,ic)-svr(ic)*w                                            3894
      dlx=max(dlx,svr(ic)**2)                                              3895
22661 continue                                                             3896
22641 continue                                                             3897
22642 continue                                                             3897
22670 continue                                                             3897
22671 continue                                                             3897
      nlp=nlp+nc                                                           3897
      dlx=0.0                                                              3898
22680 do 22681 k=1,ni                                                      3898
      if(iy(k).eq.0)goto 22681                                             3899
      jb=ix(k)                                                             3899
      je=ix(k+1)-1                                                         3899
      del=b(k,:)                                                           3899
      gkn=0.0                                                              3900
22690 do 22691 ic=1,nc                                                     3901
      u=(dot_product(r(jx(jb:je),ic),x(jb:je))-svr(ic)*xb(k))/xs(k)        3902
      gk(ic)=u+del(ic)*xv(k)                                               3902
      gkn=gkn+gk(ic)**2                                                    3903
22691 continue                                                             3904
22692 continue                                                             3904
      gkn=sqrt(gkn)                                                        3904
      u=1.0-alt*vp(k)/gkn                                                  3905
      if(u .gt. 0.0)goto 22711                                             3905
      b(k,:)=0.0                                                           3905
      goto 22721                                                           3906
22711 continue                                                             3907
      b(k,:)=gk*(u/(xv(k)+vp(k)*al2t))                                     3908
      call chkbnds1(nc,gk,gkn,xv(k),cl(1,k),cl(2,k),  vp(k)*al2t,alt*vp(   3910 
     *k),b(k,:),isc,jerr)
      if(jerr.ne.0) return                                                 3911
22721 continue                                                             3912
22701 continue                                                             3912
      del=b(k,:)-del                                                       3912
      if(maxval(abs(del)).le.0.0)goto 22681                                3913
22730 do 22731 ic=1,nc                                                     3913
      dlx=max(dlx,xv(k)*del(ic)**2)                                        3914
      r(jx(jb:je),ic)=r(jx(jb:je),ic)  -del(ic)*w(jx(jb:je))*(x(jb:je)-x   3916 
     *b(k))/xs(k)
22731 continue                                                             3917
22732 continue                                                             3917
      if(mm(k) .ne. 0)goto 22751                                           3917
      nin=nin+1                                                            3918
      if(nin .le. nx)goto 22771                                            3918
      jxx=1                                                                3918
      goto 22682                                                           3918
22771 continue                                                             3919
      mm(k)=nin                                                            3919
      m(nin)=k                                                             3920
22751 continue                                                             3921
22681 continue                                                             3922
22682 continue                                                             3922
      if(jxx.gt.0)goto 22672                                               3923
      if(dlx.lt.shr)goto 22672                                             3923
      if(nlp .le. maxit)goto 22791                                         3923
      jerr=-ilm                                                            3923
      return                                                               3923
22791 continue                                                             3924
22800 continue                                                             3924
22801 continue                                                             3924
      nlp=nlp+nc                                                           3924
      dlx=0.0                                                              3925
22810 do 22811 l=1,nin                                                     3925
      k=m(l)                                                               3925
      jb=ix(k)                                                             3925
      je=ix(k+1)-1                                                         3925
      del=b(k,:)                                                           3925
      gkn=0.0                                                              3926
22820 do 22821 ic=1,nc                                                     3927
      u=(dot_product(r(jx(jb:je),ic),x(jb:je))  -svr(ic)*xb(k))/xs(k)      3929
      gk(ic)=u+del(ic)*xv(k)                                               3929
      gkn=gkn+gk(ic)**2                                                    3930
22821 continue                                                             3931
22822 continue                                                             3931
      gkn=sqrt(gkn)                                                        3931
      u=1.0-alt*vp(k)/gkn                                                  3932
      if(u .gt. 0.0)goto 22841                                             3932
      b(k,:)=0.0                                                           3932
      goto 22851                                                           3933
22841 continue                                                             3934
      b(k,:)=gk*(u/(xv(k)+vp(k)*al2t))                                     3935
      call chkbnds1(nc,gk,gkn,xv(k),cl(1,k),cl(2,k),  vp(k)*al2t,alt*vp(   3937 
     *k),b(k,:),isc,jerr)
      if(jerr.ne.0) return                                                 3938
22851 continue                                                             3939
22831 continue                                                             3939
      del=b(k,:)-del                                                       3939
      if(maxval(abs(del)).le.0.0)goto 22811                                3940
22860 do 22861 ic=1,nc                                                     3940
      dlx=max(dlx,xv(k)*del(ic)**2)                                        3941
      r(jx(jb:je),ic)=r(jx(jb:je),ic)  -del(ic)*w(jx(jb:je))*(x(jb:je)-x   3943 
     *b(k))/xs(k)
22861 continue                                                             3944
22862 continue                                                             3944
22811 continue                                                             3945
22812 continue                                                             3945
      if(dlx.lt.shr)goto 22802                                             3945
      if(nlp .le. maxit)goto 22881                                         3945
      jerr=-ilm                                                            3945
      return                                                               3945
22881 continue                                                             3947
      goto 22801                                                           3948
22802 continue                                                             3948
      goto 22671                                                           3949
22672 continue                                                             3949
      if(jxx.gt.0)goto 22602                                               3950
22890 do 22891 ic=1,nc                                                     3951
      if((b(0,ic)-bs(0,ic))**2.gt.shr) ixx=1                               3952
      if(ixx .ne. 0)goto 22911                                             3953
22920 do 22921 j=1,nin                                                     3953
      k=m(j)                                                               3954
      if(xv(k)*(b(k,ic)-bs(k,ic))**2 .le. shr)goto 22941                   3954
      ixx=1                                                                3954
      goto 22922                                                           3954
22941 continue                                                             3956
22921 continue                                                             3957
22922 continue                                                             3957
22911 continue                                                             3958
      sc=b(0,ic)+g(:,ic)                                                   3958
      b0=0.0                                                               3959
22950 do 22951 j=1,nin                                                     3959
      l=m(j)                                                               3959
      jb=ix(l)                                                             3959
      je=ix(l+1)-1                                                         3960
      sc(jx(jb:je))=sc(jx(jb:je))+b(l,ic)*x(jb:je)/xs(l)                   3961
      b0=b0-b(l,ic)*xb(l)/xs(l)                                            3962
22951 continue                                                             3963
22952 continue                                                             3963
      sc=min(max(exmn,sc+b0),exmx)                                         3964
      sxp=sxp-q(:,ic)                                                      3965
      q(:,ic)=min(max(emin*sxp,exp(sc)),emax*sxp)                          3966
      sxp=sxp+q(:,ic)                                                      3967
22891 continue                                                             3968
22892 continue                                                             3968
      s=sum(b(0,:))/nc                                                     3968
      b(0,:)=b(0,:)-s                                                      3969
      if(jxx.gt.0)goto 22602                                               3970
      if(ixx .ne. 0)goto 22971                                             3971
22980 do 22981 j=1,ni                                                      3971
      if(iy(j).eq.1)goto 22981                                             3971
      if(ju(j).eq.0)goto 22981                                             3971
      ga(j)=0.0                                                            3971
22981 continue                                                             3972
22982 continue                                                             3972
22990 do 22991 ic=1,nc                                                     3972
      r(:,ic)=w*(y(:,ic)-q(:,ic)/sxp)                                      3973
23000 do 23001 j=1,ni                                                      3973
      if(iy(j).eq.1)goto 23001                                             3973
      if(ju(j).eq.0)goto 23001                                             3974
      jb=ix(j)                                                             3974
      je=ix(j+1)-1                                                         3975
      gj=dot_product(r(jx(jb:je),ic),x(jb:je))                             3976
      ga(j)=ga(j)+((gj-svr(ic)*xb(j))/xs(j))**2                            3977
23001 continue                                                             3978
23002 continue                                                             3978
22991 continue                                                             3979
22992 continue                                                             3979
      ga=sqrt(ga)                                                          3980
23010 do 23011 k=1,ni                                                      3980
      if(iy(k).eq.1)goto 23011                                             3980
      if(ju(k).eq.0)goto 23011                                             3981
      if(ga(k) .le. al1*vp(k))goto 23031                                   3981
      iy(k)=1                                                              3981
      ixx=1                                                                3981
23031 continue                                                             3982
23011 continue                                                             3983
23012 continue                                                             3983
      if(ixx.eq.1) go to 10880                                             3984
      goto 22602                                                           3985
22971 continue                                                             3986
      goto 22601                                                           3987
22602 continue                                                             3987
      if(kxx .le. 0)goto 23051                                             3987
      jerr=-20000-ilm                                                      3987
      goto 22522                                                           3987
23051 continue                                                             3988
      if(jxx .le. 0)goto 23071                                             3988
      jerr=-10000-ilm                                                      3988
      goto 22522                                                           3988
23071 continue                                                             3988
      devi=0.0                                                             3989
23080 do 23081 ic=1,nc                                                     3990
      if(nin.gt.0) a(1:nin,ic,ilm)=b(m(1:nin),ic)                          3990
      a0(ic,ilm)=b(0,ic)                                                   3991
23090 do 23091 i=1,no                                                      3991
      if(y(i,ic).le.0.0)goto 23091                                         3992
      devi=devi-w(i)*y(i,ic)*log(q(i,ic)/sxp(i))                           3993
23091 continue                                                             3994
23092 continue                                                             3994
23081 continue                                                             3995
23082 continue                                                             3995
      kin(ilm)=nin                                                         3995
      alm(ilm)=al                                                          3995
      lmu=ilm                                                              3996
      dev(ilm)=(dev1-devi)/dev0                                            3997
      if(ilm.lt.mnl)goto 22521                                             3997
      if(flmin.ge.1.0)goto 22521                                           3998
      me=0                                                                 3998
23100 do 23101 j=1,nin                                                     3998
      if(a(j,1,ilm).ne.0.0) me=me+1                                        3998
23101 continue                                                             3998
23102 continue                                                             3998
      if(me.gt.ne)goto 22522                                               3999
      if(dev(ilm).gt.devmax)goto 22522                                     3999
      if(dev(ilm)-dev(ilm-1).lt.sml)goto 22522                             4000
22521 continue                                                             4001
22522 continue                                                             4001
      g=log(q)                                                             4001
23110 do 23111 i=1,no                                                      4001
      g(i,:)=g(i,:)-sum(g(i,:))/nc                                         4001
23111 continue                                                             4002
23112 continue                                                             4002
      deallocate(sxp,b,bs,r,q,mm,is,sc,ga,iy,gk,del,sxpl)                  4003
      return                                                               4004
      end                                                                  4005
      subroutine psort7 (v,a,ii,jj)                                             
c                                                                               
c     puts into a the permutation vector which sorts v into                     
c     increasing order. the array v is not modified.                            
c     only elements from ii to jj are considered.                               
c     arrays iu(k) and il(k) permit sorting up to 2**(k+1)-1 elements           
c                                                                               
c     this is a modification of cacm algorithm #347 by r. c. singleton,         
c     which is a modified hoare quicksort.                                      
c                                                                               
      dimension a(jj),v(jj),iu(20),il(20)                                       
      integer t,tt                                                              
      integer a                                                                 
      real v                                                                    
      m=1                                                                       
      i=ii                                                                      
      j=jj                                                                      
 10   if (i.ge.j) go to 80                                                      
 20   k=i                                                                       
      ij=(j+i)/2                                                                
      t=a(ij)                                                                   
      vt=v(t)                                                                   
      if (v(a(i)).le.vt) go to 30                                               
      a(ij)=a(i)                                                                
      a(i)=t                                                                    
      t=a(ij)                                                                   
      vt=v(t)                                                                   
 30   l=j                                                                       
      if (v(a(j)).ge.vt) go to 50                                               
      a(ij)=a(j)                                                                
      a(j)=t                                                                    
      t=a(ij)                                                                   
      vt=v(t)                                                                   
      if (v(a(i)).le.vt) go to 50                                               
      a(ij)=a(i)                                                                
      a(i)=t                                                                    
      t=a(ij)                                                                   
      vt=v(t)                                                                   
      go to 50                                                                  
 40   a(l)=a(k)                                                                 
      a(k)=tt                                                                   
 50   l=l-1                                                                     
      if (v(a(l)).gt.vt) go to 50                                               
      tt=a(l)                                                                   
      vtt=v(tt)                                                                 
 60   k=k+1                                                                     
      if (v(a(k)).lt.vt) go to 60                                               
      if (k.le.l) go to 40                                                      
      if (l-i.le.j-k) go to 70                                                  
      il(m)=i                                                                   
      iu(m)=l                                                                   
      i=k                                                                       
      m=m+1                                                                     
      go to 90                                                                  
 70   il(m)=k                                                                   
      iu(m)=j                                                                   
      j=l                                                                       
      m=m+1                                                                     
      go to 90                                                                  
 80   m=m-1                                                                     
      if (m.eq.0) return                                                        
      i=il(m)                                                                   
      j=iu(m)                                                                   
 90   if (j-i.gt.10) go to 20                                                   
      if (i.eq.ii) go to 10                                                     
      i=i-1                                                                     
 100  i=i+1                                                                     
      if (i.eq.j) go to 80                                                      
      t=a(i+1)                                                                  
      vt=v(t)                                                                   
      if (v(a(i)).le.vt) go to 100                                              
      k=i                                                                       
 110  a(k+1)=a(k)                                                               
      k=k-1                                                                     
      if (vt.lt.v(a(k))) go to 110                                              
      a(k+1)=t                                                                  
      go to 100                                                                 
      end                                                                       