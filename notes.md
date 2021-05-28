###### ##########
# Slide 1
###### ##########

Wow, Julie, thank you for that slightly hyperbolic intro.

Thank you all for joining. I'm excited to share this work with you.

This presentation is about adapting statistical and machine learning methods for pattern identification in environmental health

###### ##########
## Slide 2
###### ##########

we'll start with a brief roadmap --
    
    i will give you some background on what even is an environmental mixture or pattern identification

    and then i will describe two methods we've been working with and how they perform on environmental data

    i will finish with an example of pattern identification as part of a two-stage health model

    ...

    we'll start with a bit of background

###### ##########
# Slide 3
###### ##########

So what exactly is an environmental mixture?

There is no strict definition in our field, though

The NIEHS has described a mixture as having at least 3 independent chemicals or chemical classes, which is a pretty conservative lower bound

but more generally, exposure to a mixture indictates multiple simultaneous stressors

--these can be chemicals, as in both examples that I'll show you
--but they can also include non chemical exposures, like socioeconomic status, diet, or built environment

###### ##########
# Slide 4
###### ##########

* in the modern era...We are exposed to hundreds (potentially thousands?) of chemicals at any given time

* epidemiological studies have traditionally analyzed chemicals independently, or one at a time, which we know doesn't account for these complex simultaneous exposures.

*   Many pollutants are correlated and some combinations of exposure are more common than others.

*   These combinations likely induces different responses Compared with single chemical exposures.

###### ##########
## Slide 5
###### ##########

Both of the mixtures of interest in this work include endocrine disrupting chemicals or EDCs

EDCs are defined by their biological mechanism as exogenous chemicals that interfere with any aspect of hormonal action

later we will talk more about EDC exposure during pregnancy which can disrupt processes critical for brain development

And exposure to endocrine disruptors is Ubiquitous through everyday consumer products, such as those on the right, and from industrial sources

They also prove difficult to study because they're often highly correlated in the environment

this reinforces the need to study them comprehensively as a mixture

###### ##########
## Slide 6
###### ##########

the importance of assessing exposure to mixtures such as these is now well-recognized

the Million dollar question: has become

How can we represent the compexity of reality in a (single) statistical model?

and we ask this because chemical mixtures introduce some challenges such as high-dimensionality and high correlations that Require flexible tools

this is why we are adjusting methods from other fields to environmental research questions

these mixtures questions generally fall under 1 of 5 topics,
    which you can see here on the right

and 

No one statistical method outperforms others for all research questions

The choice of method, then, should follow the research question

###### ##########
## Slide 7
###### ##########

in this work, we focus on pattern identification in environmental epidemiology

where identifying patterns of chemical exposure in a population
    can help to identify sources of exposure or behaviours leading to
    exposure

and linking these patterns with potential health outcomes can better

* Inform regulatory action, or public health interventions

###### ##########
## Slide 8
###### ##########

Standard methods in pattern recognition can be separated at a high level into dimension reduction or clustering methods, as you can see on the right

While these methods are used regularly, they all come with their own limitations

* first of all, the choice of patterns is subjective--
    meaning that different researchers using the same dataset and the same method may come up with a different number of underlying factors or clusers,
    and thus different results

* a second point is that Outliers may affect solution of all of these methods--where unique or extreme events may influence the patterns detected

* a third point is a problem particularly relevant in environmental health
    where chemical concentrations may be below the analytic limit of detection, or LOD, making them harder to quantify

The next two bullets are both about interpretability of results
    --first, chemical concentrations cannot be negative, 
    so factors that contain positive and negative numbers are not immediately intuitive, 
    --potentially negative chemical loadings cannot be easily interpreted as either present or abscent,
    --and potentially negative individual scores don't obviously convey exposure level

    --second, the orthogonality constraint seen most often in PCA isn't realistic in environmental mixtures where we expect patterns to overlap and we want to allow chemicals to have multiple sources

* finally, commonly used methods have No measure of uncertainty -- and this is important because 
    --results of pattern identification methods are often the first step in a two-stage process to estimate health effects of identified patterns. 
    --Without accounting for the uncertainty in the first step, we are artifically inflating our confidence in the final health model

in this work we present two methods capable of addressing some of these issues

###### ##########
## Slide 9 TOC
###### ##########

I'll start with PCP, describe the method and the work we've done to validate it, along with an application to environmental data

###### ##########
## Slide 10
###### ##########

Principal Component Pursuit was Designed in the computer vision field and I will show you an example of that in the next slide

* It Decomposes the design matrix, which in our case is an exposure matrix, into a low rank L matrix and a sparse matrix S
    
    Here is the objective function:

    ...

    * Where the nuclear norm in the highlighted first term on the L matrix forces it to be low rank
        *   We take the underlying low rank structure to represent consistent exposure patterns

    ...

    * next The L1 norm in the second term on the S matrix forces it to be sparse
        *   We take the Sparse values as unique / extreme events

        *   This also makes PCP robust to noisy or corrupt data because outlying 
            values are absorbed into the sparse matrix and don't affect
            the patterns identified in the low rank matrix

    ...

    * The final term in the objective function is to minimize the error between observed and predicted values

    ...

    * and mu and lambda are both hyper-parameters that toggle the
        contribution of the corresponding terms to the objective function

###### ##########
## Slide 11
###### ##########

here is an example from  computer vision using security camera footage

on the left is the original data of individuals walking through a lobby.

we applied PCP to this data, and the estimated L and S matrices are in the center and right hand columns

the L matrices contain the parts of the images that don't change, the background of the lobby

-- these are the consistent patterns which correspond to patterns of environmental exposures in our work

the S matrices contain individuals walking across the frame, which are unique events,
    corresponding to extreme chemical exposures in our field

###### ##########
## Slide 12
###### ##########

in addition to introducing PCP as a tool for pattern recognition, we've also adapted it to environmental data in several ways.

    1. We included a formulation of the objective function where the hyper-parameters dont need to be tuned, taking this burden off of the researcher

    2. We have relaxed the nuclear norm to a non-convex rank penalty that performs better on environmental data

    3.  we have added a non-negativity constraint onto the low rank matrix to help better identify patterns

    4. To address the presence of values <LOD, we have included novel penalties in the objective function. these use the low rank structure across the exposure matrix as a whole to impute values <LOD.

    * These extensions are designed to preserve the robust nature of PCP, while adapting it to specific hurdles we face in environmental health.

this work was done in collaboration with John Wright and his students at the data science institute, along with jeff goldsmith in biostats

next, I tested this extended pcp which we're calling PCP-LOD on simulations and applied it to environemntal health data

###### ##########
## Slide 13
###### ##########

in environmental mixtures and environmental health more generally, we can never know the true underlying data generating process, but with simulations we can

so the goal here was to generate examples that look like real environmental data, but where we know the true structure 

to do this, i modeled simulations using summary statistics like mean, median, variance, covariance, and skew of real multi-pollutant exposures

To simulate the low rank structure,
    * i generate individual scores from indendendent lognormal distributions
    --these scores express the magnitude of an individual's exposure to a given pattern

    * & i generated chemical loadings as you see on the right, where some chemicals load uniquely on one pattern, and some load across two patterns
    --loadings define the composition of the patterns, so they tell us which chemicals make up the pattern and how much they contribute

    * all simulations have 4 underlying patterns

    * next I added noise in one of three ways and artifically created values below the LOD.

    --when I say noise, I mean the random irregularity we find in any real life data, like measurement error or sampling error. so I added it to make the structure more realistic.

...

here, we have one example of simulated chemicals. their moderate to high- correlation structure pretty well represents what we would have expected in real life

###### ##########
## Slide 14
###### ##########

In this and the next slide, we compared PCP-LOD performance with
    PCA where values <LOD were imputed as LOD/sqrt(2)

here we see the overall error for each method

on the y axis, we have relative l2 predictive error, which is relative to the underlying simulation matrix before noise was added
    * lower error indicates better performance

on the x axis, we have increasing proportion of values <LOD

The panels show results for different structures of added noise. 

and boxplots are colored according to method--
with PCA in orange,
and PCP-LOD in blue

we see that PCP-LOD consistently outperformed PCA in the left and center panels when noise was low and when sparse events were added

but PCA performed better in the rightmost panel with high levels of noise and 75% of the data <LOD where there was less structure across the matrix for PCP-LOD to borrow from.

###### ##########
## Slide 15
###### ##########

next we see relative prediction error again, this time stratified by LOD

so the axes and the coloring are the same

and we have values > LOD in the top panels and values <LOD in the bottom panels

we see for both methods, that poorer performance on values <LOD really drove the increasing overall error in the last slide.

With these simulations, we wanted to highlight that PCP-LOD outperforms PCA in many situations, when error is low and when there are, in fact sparse events, and even when error is high, as long as enough of the exposures are above the LOD

###### ##########
## Slide 16
###### ##########

we will transition now to an application of PCP-LOD on environmental data

we used data from The National Health and Nutrition Examination Survey (NHANES) in their 2001-2002 cycle. 

NHANES is a nationally representative sample of noninstitutionalized Americans.

importantly for this analysis, it includes biospecimens which are analyzed for certain environmental exposures

in this work, we were interested in a mixture of 35 polychlorinated biphenols or PBCs, dioxins, and furans which are all persistant organic pollutants (POPs)

We wanted to identifty patterns of exposure and to determine if identified patterns correspond with know sources or behaviors contributing to exposure. 

###### ##########
## Slide 17
###### ##########

We first defined our mixture as the chemicals detected in at least 50% of samples. this included 14 PCBs, 4 furans and 3 dioxins which you can see here to the right of the dotted line. with these 21 POPs, about 25% of the entire mixture was below the LOD

###### ##########
## Slide 18
###### ##########

this is the correlation matrix of included POPs. The darker the red, the higher the correlation. All chemicals were positively correlated, as you can see, with some correlations being quite high

we applied PCP-LOD to this exposure matrix to identify underlying patterns

###### ##########
## Slide 19
###### ##########

PCP-LOD returned a rank 3 low rank solution, described here

each panel represents a principal component, the chemicals are on the y axis, their loadings on each component are on the x axis and they are colored according to chemical class.

The loadings can be understood as the weights for each chemical on each component

so The first component on the left describes overall exposure because all of the chemicals load in the same direction

the second component separated dioxins and furans, which are generally more toxic, in the negative direction, from PCBs in the positive direction,

and the third component separated higher molecular weight PCBs in the positive direction from lower molecular weight PCBs in the negative direction,

Depending on the research question, any or all of these components could be included in subsequent analyses with relevant health outcomes.

###### ##########
## Slide 20
###### ##########

this next figure depicts extreme or unique events. chemicals are along the x axis with participants on the y axis, and each chemical per participant is colored white for no event, blue for an extremely low event, and red for an extremely high event.

the first thing you notice is a large patch of white in the middle. 44% of participants had no unusual events, meaning that all of their exposure was explained by the 3 patterns in the last slide.

The remaining individuals had  at least one high or low event left unexplained by patterns and captured in the sparse matrix, where about 6% of all entries where non-zero.

these unique observations did not affect the formation of the identified patterns and we retained them for potential further analysis.

###### ##########
## Slide 21
###### ##########

this is a quick recap slide before we switch topics.

so the Big picture concepts about PCP-LOD are:

* that it identifies an underlying low rank structure that is not influenced by outliers
* these outliers are not discarded, as we saw in the previous slide
* PCP-LOD outperformed PCA across noise structures as long as the proportion <LOD wasnt too high
* and finally, we've removed parameter tuning, and the researcher does not need to choose the number of patterns

###### ##########
## Slide 22 TOC
###### ##########

we will transition here to bayesian nonparametric nonnegative matrix factorization or BNMF, again I'll describe the method and the simulations we've done to test it, along with an application to environmental data

###### ##########
## Slide 23
###### ##########

this is a visualization of BNMF, which is a Parts-based representation of the data where

we decompose the nxp observed data, where n is the number of participants in the study and p is the number of chemicals in the mixture into three matrices:

1. the first is an nxk matrix of Individual scores across k patterns -- 
--this contains the dimension-reduced representation for each individual

3. the last is a kxp matrix of chemical loadings -- 
--these are shared by all individuals and pick out common exposure patterns in the population

We've placed non-negative continuous Gamma priors on both of these

2. the central kxk diagonal matrix has a sparse gamma prior that shrinks out unnecessary rows of the loading matrix and corresponding columns of the score matrix. (this is the non-parametric part)
--the number of patterns k is initialzed to = the number of chemicals in the dataset, and the model infers the appropriate k from the data, so that the final estimated number of patterns << the number of chemicals

* because we're working in a Bayesian framework, our results include distributions for each entry in the three solution matrices instead of point estimates. 
--and we use these distributions to construct variational confidence intervals for all estimated values.

###### ##########
## Slide 24
###### ##########

here, I've simulated data again modeled after real multi-pollutant exposures

to test BNMF's performance compared with other methods

To create the low rank structure,
    * i again simulated individual scores from indendendent lognormal distributions

    * but I generated chemical loadings differently with increasingly complex structures.
        * as you see on the right, the simplest loading structure has every chemical loading distinctly on a single pattern
        * next, you see half of the chemicals loading distinctly and half loading across two patterns, 
        * and finally, we have all chemicals loading across two patterns

    * all have 4 underlying patterns and increasing levels of noise

and we compared BNMF performance with PCA, factor analysis, and frequentist NMF on all simulations

    ...

now on the right we have an example correlation matrix-- where chemicals are again moderately to highly correlated

In the following slides, I will show you the experimental results

###### ##########
## Slide 25
###### ##########

Here, I'm comparing BNMF with four common pattern recognition methods in terms of predictive accuracy

the key benefits of the bayesian approarch, though, isnt improved accuracy 
it's the ability to select the appropriate number of patterns and to quantify uncertainty in estimate

we wanted to make this comparison though, to demonstrate that BNMF doesn't suffer in terms of prediction to compensate for improvements in other areas

so we have relative predictive error on the y axis comparing predicted values from each model with the underlying truth before noise was added

The two panels show results for increasing amounts of added noise

on the x axis of each, we have boxplots colored according to method--
WITH BNMF IN RED
Factor Analysis in blue,
NMF with an L2 penalty in yellow
NMF with a Poisson likelihood in green
and PCA in purple

we see right away that factor analysis has the poorest performance,

PCA has the most variation, 

and, overall, non-negative methods do a better job of reconstructing the true data

the big take away here is that our method performs at least as well as some commonly used methods, though traditional nmf achieves slightly lower relative error, it lacks BNMF's ability to choose the optimal number of patterns and to provide estimates of uncertainty

###### ##########
## Slide 26
###### ##########

we see  this uncertainty quantification here.

this figure shows BNMF's coverage on simulated data.

on the x axis, each square represents increasing complexity in the underlying patterns, ranging from 10 where every chemical is distinct to one pattern to zero where every chemical loads across two patterns.

on the y axis, we have the noise level as a proportion. zero indicates that no noise was added, increasing to noise equal to half the true variability in the data

each square is colored by median coverage, which is how often the true simulated score was included within BNMF's 95% confidence interval.

we see that when there is at least one chemical that loads distinctly on each pattern and noise is below 0.4, we get approximately nominal coverage, meaning that ~95% or more of true values fall within the 95% confidence intervals

as noise increases, coverage decreases, but it still provides more information than a single point estimate

###### ##########
## Slide 27
###### ##########

here i want to take a closer look at what a confidence interval tells us

on the y axis we have 4 randomly selected simulated study participants, and we have their estimated pattern scores on the x axis.

the scores are like concentrations of the estimated patterns

each color represents a different pattern, and each distribution has a mean and 95% confidence interval as black lines over the distribution

the dashed lines represent the true simulated scores for the corresponding color

for A and B, our model is quite sure that patterns 1 and 3 respectively are much higher than the rest

for C, all distributions overlap quite a bit, but we are sure that these scores are within this range

for D, though, the true scores for patterns 1 and 3, fall outside the 95% confidence interval

and this results in lower coverage, but we can still see clearly that the true scores are contained within the entire estimated distribution, just farther out in the tail, which again does give us more information than a single point

###### ##########
## Slide 28
###### ##########

we'll transition here to another application
this data come from columbia's center for children's environmental health, which has followed a prospective cohort of mothers and children over time.

the research question is again about endocrine disrupting chemicals

We want to know if BNMF can identify interpretable exposure patterns in pregnant women?

And then we want to determine whether identified exposure patterns are associated with child intelligence

###### ##########
## Slide 29
###### ##########

Here we have correlations of phenols and phthalate metabolites measured in pregnant women.

and they are clearly much lower than those in the previous mixture

though there are some notably high correlations like in the bottom left among metabolites of a single parent compound

generally, the correlations here are low to moderate

###### ##########
## Slide 30
###### ##########

after applying BNMF to this mixture we found 2 underlying patterns.

each chemical on the x axis is colored according to its weight on each pattern, and the chemicals are grouped with phenols on the left and phthalates on the right

the first pattern shows mostly phthalates + one phenol, BPA on the far left

the second pattern shows mostly phenols, with one phthalate, MEP, in the center

###### ##########
## Slide 31
###### ##########

in the next section we include these patterns in a health model, so I wanted to take a minute to look at them more qualitatively

here we have two word clouds representing each pattern.
the chemicals are colored according to their class, with phthalate metabolites in blue and phenols in orange

and the size of the word corresponds with the strength of its loading on the pattern.

both phthalates and phenols are found in food packaging and personal care products, so they arent distinct groupings, several of these chemicals appear in both clouds.

For pattern 1, higher molecular weight phthalates are more likely to be found in food packaing, as is BPA, so we're interpreting it as a Potentially shared route through diet

for pattern 2, phenols, in general, are often found in personal care products, as is the parent compound DEP of this particular phthalate MEP which is often found in products with fragrance
• so there is a Potentially shared route through personal care products

the take away here is that BNMF has successfully identified interpretable environmental exposures patterns which may be later linked to health outcomes

###### ##########
## Slide 32
###### ##########

I want to pause to emphasize the novelty of BNMF.
* the parts based representation makes the solution much more interpretable

we've removed the subjective choice of the number of patterns

and the variational confidence interval accounts for uncertainty in BNMF which we can propagate into a health model, like in the next example.

###### ##########
## Slide 33
###### ##########

here I have a couple of background slides for this application and a health analysis of BNMF-identified patterns in relation to child cognition

###### ##########
## Slide 34
###### ##########

Chemical exposure is unique to each individual and varies across the lifecourse

Pregnancy, especially, is a critical window of exposure because The developing fetal brain is uniquely susceptible to environmental insults.

* Neural development begins within weeks of conception and the fetal brain develops rapidly, 

* So Even minor stressors can dramatically affect the precisely regulated steps critical for proper development. 

in utero exposure to environmental toxicants may lead to adverse health outcomes in both the short and long-term 

and there is substantial evidence linking Prenatal environmental exposures with cognitive deficits in children

###### ##########
## Slide 35
###### ##########
 
While a small decrease in intellence may not be clinically relevant in a single individual, This slide gives a general example of population-level effects of even small cognitive deficits

We see two density distributions for IQ as a global measure of cognition, we have an unexposed population in blue and an exposed population in gray.

The dashed vertical line is the threshold for IQ scores consistent with intellectual disabilities (IQ <70)

In the unexposed population, the mean IQ is 100, and in the exposed population the mean IQ is 95. This five-point shift in IQ results in nearly a doubling in the proportion of children with IQ scores consistent with intellectual disabilities in the exposed population compared with the unexposed population ((4.48% and 2.27%, respectively))

this really emphasizes that, Even a small downward shift in cognition may have a substantial population impact.

###### ##########
## Slide 36
###### ##########

* the outcome of interest here is child IQ at age 7 measured with the Wechsler Intelligence Scale for Children, which is a validated measure of a child's general intellectual performance

* we use the EDC patterns that I just describes as the exposures of interest

* and we include the listed covariates as potential confounders or precision variables

* all models include an interaction term between child sex and pattern exposure 

###### ##########
## Slide 37
###### ##########

we ran two types of models--traditional multivariable regression and a Bayesian hierarchical model

the model at the top describes both versions, where Yi is child IQ, Zi is pattern exposure, Si is child sex, Zi times Si is the interaction term, and Xi includes all covariates

In the traditional regression, we took the expected value or the mean of the distribution over individual scores, where a is the variable that chose the optimal number of patterns, and W times a is the scaled score 

In the Bayesian model, we included the entire distribution for each individual score as the product of the gamma distribution over W and the gamma distribution over a

while the top level equation looks the same the both, the bayesian model includes all information on estimated distributions.

###### ##########
## Slide 38
###### ##########

this figure shows associations between identified patterns and child IQ, stratified by sex

with the first pattern on the left and the second pattern on the right

the points show the beta coefficients with their 95% confidence intervals

with circles representing the association in females and triangles representing the association in males

they're colored according to model, with the Bayesian model in orange and the traditional model in blue

the traditional and bayesian models agree on all accounts
while the bayesian confidence interval is always slightly wider, though you have to look closely to see it
this incoporates our confidence in the pattern identification step

the second pattern on the right was not associated with child IQ in either males or females

and the first pattern was not associated with IQ in males

we did find an association between the the first pattern and child IQ in females, and you can see that the confidence intervals for those beta coeffients don't include zero

we found that a 
One standard deviation increase in this pattern was associated with a decrease of 3.4 IQ points, on average, in female children

###### ##########
## Slide 39
###### ##########

to bring this part to a close, the identified pattern of phthalate and BPA during pregancy was associated with lower IQ in females

and the findings from the bayesian model reinforce those from the tradional model

we saw an average loss in females of 3.4 IQ points with every unit increase in this pattern. 3.4 lost IQ points in any particular child is unlikely to have clinically relevant or noticable effects, but it may have substantial impact at a population level

further, EDC exposure is largely modifiable, meaning that preventative strategies can minimize or block it.

and in some circumstances it can be easier to target patterns than individual chemicals for public health or regulatory action 

###### ##########
## Slide 40
###### ##########

Introducing two pattern recognition techniques really begs the question: what's the difference? When would you use one or the other? 

And you'll notice that none of this work directly compared PCP-LOD and BNMF. This is because we consider both methods to be works in progress, and we wanted to compare them with more standard methods in the field.

When to use one over the other, though, will depend on both the specific research question and the data at hand, and there will be some overlap where either method is appropriate.

For example, I would choose PCP-LOD if I expected extreme events or if a large number of measurements were below the LOD.

However, PCP-LOD needs to be paired with another method to produce individual scores and chemical loadings similar to those obtained from BNMF, so the appropriateness of the second approach would also affect the decision. 

I would use BNMF if I required uncertainty quantification for any reason, whether to include in a two-stage health model or to provide confidence intervals on scores or loadings.

These are general guidelines though and not really hard rules because every study has different elements that can influence the choice of analytical method.

###### ##########
## Slide 41
###### ##########

As with any research, ours comes with certain challenges

but, since we consider both methods to be works-in-progess, these challenges present opportunities for next steps and future methodological developments

For PCP, I listed the fact that it doesn't directly estimate loadings and scores as a limitation, and including this estimation in the same step as the decomposition into low rank and sparse components would be a natural extenson

we also anticipate future work with the sparse solution matrix. In the NHANES example, I showed you how PCP-LOD identified high and low extreme events that weren't explained by the patterns and separated them for further study. this is a real strength of the method, but we don't know what exactly this 'further study' should look like. In one application, we included sparse events with identified patterns as exposures in a penalized regression with a health outcome, so that coefficients for extreme events unrelated with the outcome would be pushed to zero. but this is just one potential example within a range of possibilities. 

for BNMF, we saw that the coverage of the 95% variational confidence intervals got worse as the simulations became noisier. this is expected, but environmental health data always comes with some noise from measurement error or sampling error.

BNMF as it is currently written approximates the posterior distribution using variational inference. this approach has a lot of benefits, but it is possible that a full Markov chain Monte Carlo to take draws from the true posterior would provide improved coverage in the presence of noise. 

There is also room for future work to reformulate the two-stage hierarchical model as a fully supervised model where the outcome of interest informs the `grouping' of the chemicals in the mixture. A supervised approach could provide insight on potential biological pathways, and it could make identified patterns more similar across populations.

A final next step for both methods is that we plan to develop and share user-friendly statistical software packages in R 

we will include detailed documentation and vignettes for guidance on proper use, inputs, interpretation, and limitations of both methods.

we want to provide this resource so that other researchers assessing exposures to mixtures can easily apply PCP-LOD or BNMF

###### ##########
## Slide 42
###### ##########

to wrap this up, no statistical method can answer all research questions, but here i've presented two methods to answer questions concerning pattern identification in public health.

these methods included work to remove the researcher from pattern selection.

to emphasize the interpretability of results

and to account for uncertainty in estimation

these methods can contribute to the growing field of environmental mixtures research

###### ##########
## Slide 43
###### ##########




