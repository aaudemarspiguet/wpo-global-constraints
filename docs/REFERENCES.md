# References

## The paper and the patent

- **[P]** Weicong Ding, Dinesh Govindaraj, S V N Vishwanathan. 2019. *Whole Page
  Optimization with Global Constraints.* KDD '19, pp. 3153–3161.
  https://doi.org/10.1145/3292500.3330675
- **[X10]** Weicong Ding, Dinesh Govindaraj, Vishwanathan Swaminathan (inventors);
  Amazon Technologies, Inc. *Optimizing content engagement with imposed content
  constraints.* US Patent 10,915,336 B1. Filed Nov. 5, 2018; granted Feb. 9, 2021.

## Reading order

If you only read five things besides the paper:

1. [X2] Krause & Golovin, sections 1–3: submodularity and why greedy works.
2. [X3] Boyd & Vandenberghe, chapter 5 (5.1–5.5): Lagrangian duality.
3. [X4] Bishop, section 3.3: Bayesian linear regression (Eq. 4).
4. [1] Agarwal et al. 2015: constrained homepage optimization, the closest prior system.
5. [24] Shah et al. 2017: a primal-dual online ranking algorithm with constraints.

## The KDD paper's bibliography (numbering as in the paper)

The "Used in" column says where this guide relies on each work. Descriptions
follow how the KDD paper positions them; read the originals before citing them
yourself.

| # | Reference | Used in |
|---|---|---|
| 1 | Deepak Agarwal, Shaunak Chatterjee, Yang Yang, Liang Zhang. 2015. Constrained optimization for homepage relevance. WWW '15, 375–384. | §1.2, §2.3, §5.6–5.7 |
| 2 | Deepak Agarwal, Bee-Chung Chen, Pradheep Elango. 2009. Spatio-temporal models for estimating click-through rate. WWW '09, 21–30. | §1.4 (pointwise relevance) |
| 3 | Shipra Agrawal, Nikhil Devanur. 2016. Linear contextual bandits with knapsacks. NeurIPS, 3450–3458. | §2.3 |
| 4 | Amr Ahmed, Choon Hui Teo, S.V.N. Vishwanathan, Alex Smola. 2012. Fair and balanced: learning to present news stories. WSDM '12, 333–342. | §2.3, §3.6 (submodular diversity) |
| 5 | K. Basu, S. Chatterjee, A. Saha. 2016. Constrained multi-slot optimization for ranking recommendations. arXiv:1602.04391. | §2.3 |
| 6 | Kinjal Basu, Ankan Saha, Shaunak Chatterjee. 2017. Large-scale quadratically constrained quadratic program via low-discrepancy sequences. NeurIPS, 2297–2307. | §1.4 |
| 7 | Laming Chen, Guoxin Zhang, Hanning Zhou. 2018. Improving the diversity of top-N recommendation via determinantal point process. NeurIPS. | §1.1, §1.4, §3.6 |
| 8 | Ye Chen, Weiguo Liu, Jeonghee Yi, Anton Schwaighofer, Tak W. Yan. 2013. Query clustering based on bid landscape for sponsored search auction optimization. KDD '13, 1150–1158. | §2.3, §5.4, §5.7 |
| 9 | K. Christakopoulou, J. Kawale, A. Banerjee. 2017. Recommendation with capacity constraints. CIKM '17, 1439–1448. | §1.2, §2.4 (penalty-in-the-loss alternative) |
| 10 | C. Clarke, M. Kolla, G. V. Cormack, O. Vechtomova, A. Ashkan, S. Büttcher, I. MacKinnon. 2008. Novelty and diversity in information retrieval evaluation. SIGIR '08, 659–666. | §1.4 |
| 11 | Paul Covington, Jay Adams, Emre Sargin. 2016. Deep neural networks for YouTube recommendations. RecSys '16, 191–198. | §1.4 |
| 12 | A. Das, M. Datar, A. Garg, S. Rajaram. 2007. Google News personalization: scalable online collaborative filtering. WWW '07, 271–280. | §1.4 |
| 13 | Nikhil R. Devanur, Jugal Garg, Ruta Mehta, Vijay V. Vazirani, Sadra Yazdanbod. 2018. A new class of combinatorial markets with covering constraints: algorithms and applications. SODA '18, 2311–2325. | §5.6–5.7 |
| 14 | Nikhil R. Devanur, Zhiyi Huang, Nitish Korula, Vahab S. Mirrokni, Qiqi Yan. 2016. Whole-page optimization and submodular welfare maximization with online bidders. ACM TEAC 4(3). | §1.1, §1.4, §3.6 |
| 15 | Thore Graepel, Joaquin Quiñonero Candela, Thomas Borchert, Ralf Herbrich. 2010. Web-scale Bayesian click-through rate prediction for sponsored search advertising in Microsoft's Bing search engine. ICML. | §3.5 (message passing), §6.3 |
| 16 | R. Gupta, G. Liang, H. Tseng, R. K. Holur V., X. Chen, R. Rosales. 2016. Email volume optimization at LinkedIn. KDD '16, 97–106. | §5.4, §5.6–5.7 |
| 17 | X. He, J. Pan, O. Jin, T. Xu, B. Liu, T. Xu, Y. Shi, A. Atallah, R. Herbrich, S. Bowers. 2014. Practical lessons from predicting clicks on ads at Facebook. ADKDD '14, 1–9. | §1.4, §3.5 |
| 18 | Daniel N. Hill, Houssam Nassif, Yi Liu, Anand Iyer, S.V.N. Vishwanathan. 2017. An efficient bandit algorithm for realtime multivariate optimization. KDD '17, 1813–1821. | §1.4, §6.3, §7.5 (MVT baseline) |
| 19 | Jim C. Huang, Rodolphe Jenatton, Cédric Archambeau. 2016. Online dual decomposition for performance and delivery-based distributed ad allocation. KDD '16, 117–126. | §2.3, §5.4, §5.7 |
| 20 | Neil Hurley, Mi Zhang. 2011. Novelty and diversity in top-N recommendation: analysis and evaluation. ACM TOIT 10(4). | §1.4 |
| 21 | Rodolphe Jenatton, Jim Huang, Cédric Archambeau. 2016. Adaptive algorithms for online convex optimization with long-term constraints. ICML, 402–411. | §2.3, §5.7 |
| 22 | Mehrdad Mahdavi, Rong Jin, Tianbao Yang. 2012. Trading regret for efficiency: online convex optimization with long term constraints. JMLR 13, 2503–2528. | §5.7, §11 |
| 23 | Georgina Peake, Jun Wang. 2018. Explanation mining: post hoc interpretability of latent factor models for recommendation systems. KDD '18, 2060–2069. | §1.1 (out of scope: grouping content into widgets) |
| 24 | Parikshit Shah, Akshay Soni, Troy Chevalier. 2017. Online ranking with constraints: a primal-dual algorithm and applications to web traffic-shaping. KDD '17, 405–414. | §2.3, §5.7, §11 |
| 25 | Wen Sun, Debadeepta Dey, Ashish Kapoor. 2017. Safety-aware algorithms for adversarial contextual bandit. ICML, 3280–3288. | §2.3 |
| 26 | Choon Hui Teo, Houssam Nassif, Daniel Hill, Sriram Srinivasan, Mitchell Goodman, Vijai Mohan, S.V.N. Vishwanathan. 2016. Adaptive, personalized diversity for visual discovery. RecSys '16, 35–38. | §1.4, §3.6, DIV baseline throughout |
| 27 | Yue Wang, Dawei Yin, Luo Jie, Pengyuan Wang, Makoto Yamada, Yi Chang, Qiaozhu Mei. 2016. Beyond ranking: optimizing whole-page presentation. WSDM '16, 103–112. | §1.4, §7.5 (MVT) |
| 28 | M. Wilhelm, A. Ramanathan, A. Bonomo, S. Jain, E. Chi, J. Gillenwater. 2018. Practical diversified recommendations on YouTube with determinantal point processes. CIKM '18, 2165–2173. | §1.1, §2.3 |
| 29 | Hao Yu, Michael Neely, Xiaohan Wei. 2017. Online convex optimization with stochastic constraints. NeurIPS, 1427–1437. | §1.2, §2.3, §5.7, §11 |

## Outside references added by this guide

| # | Reference | Used in |
|---|---|---|
| X1 | G. L. Nemhauser, L. A. Wolsey, M. L. Fisher. 1978. An analysis of approximations for maximizing submodular set functions, I. *Mathematical Programming* 14, 265–294. | §6.1 (the $1 - 1/e$ greedy bound) |
| X2 | Andreas Krause, Daniel Golovin. 2014. Submodular function maximization. In *Tractability: Practical Approaches to Hard Problems*, Cambridge University Press. | §3.2 |
| X3 | Stephen Boyd, Lieven Vandenberghe. 2004. *Convex Optimization*, Ch. 5 "Duality". Cambridge University Press. Free PDF on the authors' site. | §5.2 |
| X4 | Christopher M. Bishop. 2006. *Pattern Recognition and Machine Learning*, §3.3 "Bayesian linear regression". Springer. | §3.5 |
| X5 | Nick Craswell, Onno Zoeter, Michael Taylor, Bill Ramsey. 2008. An experimental comparison of click position-bias models. WSDM '08. | §3.1 (cascade assumption) |
| X6 | F. Maxwell Harper, Joseph A. Konstan. 2015. The MovieLens datasets: history and context. *ACM TiiS* 5(4), 19:1–19:19. https://doi.org/10.1145/2827872 | DATASETS §2 |
| X7 | Fernando B. Pérez Maurera, Maurizio Ferrari Dacrema, Lorenzo Saule, Mario Scriminaci, Paolo Cremonesi. 2020. ContentWise Impressions: an industrial dataset with impressions included. CIKM '20, 3093–3100. https://doi.org/10.1145/3340531.3412774 | DATASETS §3 |
| X8 | Follow-up work from Politecnico di Milano / ContentWise on offline evaluation for multi-carousel interfaces, e.g., the IIR 2021 workshop paper at https://ceur-ws.org/Vol-2947/paper4.pdf. Find the current version and full citation before citing. | §3.1, DATASETS §3 |
| X9 | Olivier Chapelle, Lihong Li. 2011. An empirical evaluation of Thompson sampling. NeurIPS. | §6.3 |
| X11 | Ron Kohavi, Diane Tang, Ya Xu. 2020. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing.* Cambridge University Press. | §8 |
