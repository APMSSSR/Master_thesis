import numpy as np

# intersection computation
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

# to populate group distributions
def get_pmf(cdf):
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size-1):
        pis[score+1] = cdf[score+1] - cdf[score]
    return pis

# to calculate new shifted score distributions for different bank types
def get_shifted_score_distributions(pis, shift, iterations):
    for k in range(0, iterations):
        pis_change = np.zeros(2)
        check = np.zeros(2)
    
        if shift < 0:
            for i in range(len(pis[0]) - 1, -1, -1):
                for j in range(0, len(pis)):
                    pis[j][i] += pis_change[j]
                    pis_change[j] = (pis[j][i]-pis_change[j])*np.abs(shift)
                    if i > 0:
                        pis[j][i] -= pis_change[j]
                    check[j] += pis[j][i]
            print(check)

        elif shift > 0:
            for i in range(0, len(pis[0])):
                for j in range(0, len(pis)):
                    pis[j][i] += pis_change[j]
                    pis_change[j] = (pis[j][i]-pis_change[j])*np.abs(shift)
                    if i < len(pis[0])-1:
                        pis[j][i] -= pis_change[j]              
                    check[j] += pis[j][i]
            print(check)
    
    return pis

def pis2cdf(pis):
    cdf = np.zeros(pis.size)
    cumulation = 0
    for i in range(cdf.size):
        cumulation += pis[i]
        if cumulation > 1:
            cdf[i] = 1
        else:
            cdf[i] = cumulation

    return cdf


#TODO> rewrite for multiple banks
#distribution if we take into account that customers take loan with lowest interest rate
def get_pi_combined(pi_normal,pi_conservative, scores, score_interest_intersect):
    pis = np.zeros(pi_conservative.size)
    #find index of scores where the two interest rates change
    change_index = 0
    for i in range(pi_normal.size-1):
        if scores[i] < score_interest_intersect and scores[i+1] > score_interest_intersect:
            change_index = i
            print(change_index, scores[i], scores[i+1])
            break
    
    #add all the distribs of those getting loan at conservative bank for lower interest
    cumulative_cdf = 0
    for i in range(pi_conservative.size-1,1,-1):
        if i > change_index:
            pis[i] = pi_conservative[i]
            cumulative_cdf += pi_conservative[i]
    
    print(cumulative_cdf)
    
    rest_index = 0
    cumulative_cdf_normal = 0
    #find index up to cumulative cdf of normal bank
    for i in range(pi_normal.size-1):
        cumulative_cdf_normal += pi_normal[i]
        if cumulative_cdf_normal + pi_normal[i+1]  > 1-cumulative_cdf:
            rest_index = i+1
            print(rest_index, cumulative_cdf_normal+ pi_normal[i+1])
            break

    #add those pi to combined pis
    for i in range(pi_conservative.size-1):
        if i <= rest_index:
            pis[i] += pi_normal[i]
    #check
    cumulative_check = 0
    for i in range(pi_conservative.size-1):
        cumulative_check += pis[i]
    print(cumulative_check)
        
    
    return pis
