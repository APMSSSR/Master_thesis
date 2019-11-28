import numpy as np
import random
import matplotlib.pyplot as plt

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


#get score mapping to interest rates for different banks
def get_i_rates(interest_rates, score_range, bank_names):
    plt.figure(0)
    x_axis = np.linspace(score_range[0],score_range[1],score_range[1]-score_range[0]+1, dtype=int)
    tmp_rates = []
    for i in range(len(interest_rates)):
        coefficients = np.polyfit(score_range, interest_rates[i], 1)
        polynomial = np.poly1d(coefficients)
        y_axis = polynomial(x_axis)
        tmp_rates.append(y_axis)
        plt.plot(x_axis, y_axis,label=bank_names[i]+" bank")
    
    #plt.plot(x_axis, np.max(tmp_rates)-np.amin(tmp_rates,axis=0),color='black',LineStyle=':', label="applicant utility curve")
    plt.ylabel('Interest rate')
    plt.xlabel('Score')
    plt.title('Dependence of interest rate on score for different banks')
    plt.grid('on')
    plt.legend(loc="lower left")
    plt.show()
  
    score_interest_rates = dict(zip(x_axis, np.transpose(tmp_rates)))
    return score_interest_rates


#get simulated repay outcome of score probability(1=repaid, 0=default )
def get_repay_outcome(repay_probability):
    random_number = random.random()
    outcome = 1
    #print(random_number)
    if random_number < 1-repay_probability:
        outcome = 0
    return outcome


#outputs the index of a bank with lowest interest rate 
def get_minimal_interest_bank(score_interest_rates, applicant_scores, N_banks, group_index, applicant_index):
    interest_values=[]
    for bank in range(N_banks):
        interest_values.append(score_interest_rates[applicant_scores[group_index][bank][applicant_index]][bank])
    
    return interest_values.index(min(interest_values)) 

def get_next_ref_applicant_scores(ref_applicants, score_interest_rates, applicant_scores, MP_max_rate, loan_repaid_probs, N_groups, N_banks, score_change_repay, score_change_default, score_range):
    next_ref_applicant_scores = []
    for i in range(0, N_groups):
        next_ref_applicant_scores.append(np.zeros(len(ref_applicants[i])))

        for j in range(len(ref_applicants[i])-1,-1,-1):
            selection_rate = 1-j/len(ref_applicants[i])

            if selection_rate <= MP_max_rate[i][-1]:
                optimal_bank = get_minimal_interest_bank(score_interest_rates,applicant_scores,N_banks, i, j)
                
                for k in range(0,len(MP_max_rate[i])):    
                    if selection_rate <= MP_max_rate[i][k] and optimal_bank == k:
                        if get_repay_outcome(loan_repaid_probs[i](ref_applicants[i][j])):
                            next_ref_applicant_scores[i][j]=ref_applicants[i][j] + score_change_repay
                        else:
                            next_ref_applicant_scores[i][j]=ref_applicants[i][j] + score_change_default
                        break
                    elif selection_rate > MP_max_rate[i][k] and optimal_bank == k:
                        optimal_bank = optimal_bank + 1
            else:
                next_ref_applicant_scores[i][j]=ref_applicants[i][j]

        next_ref_applicant_scores[i][next_ref_applicant_scores[i] < score_range[0]] = score_range[0]        
        next_ref_applicant_scores[i][next_ref_applicant_scores[i] > score_range[1]] = score_range[1]
        next_ref_applicant_scores[i]=np.sort(next_ref_applicant_scores[i])
    return next_ref_applicant_scores


# to populate group distributions
def get_pmf(cdf):
    pis = np.zeros(cdf.size)
    pis[0] = cdf[0]
    for score in range(cdf.size-1):
        pis[score+1] = cdf[score+1] - cdf[score]
    return pis


# to populate multiple group distributions
def get_pmfs(cdfs):
    pis=[]
    for i in range(0,len(cdfs)):
        tmp_pis = np.zeros(len(cdfs[i]))
        tmp_pis[0] = cdfs[i][0]
        for score in range(len(cdf[i])-1):
            tmp_pis[score+1] = cdfs[i][score+1] - cdfs[i][score]
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

# get reference applicant scores
def get_ref_applicants(applicant_totals, pis_total, scores_list):
    ref_applicants = []
    for i in range(0,len(pis_total)):
        pointer = 0
        ref_applicants.append(np.zeros(applicant_totals[i]))
        for j in range(0, len(pis_total[i])):
            diff_up = 0
            diff_down = 0
            if pis_total[i][j] == 0:
                step = 0
            else:
                if j == 0:
                    diff_up = (scores_list[j+1]-scores_list[j])/2
                    step = diff_up/pis_total[i][j]
                elif j == len(pis_total[i])-1:
                    diff_down = (scores_list[j]-scores_list[j-1])/2
                    step = diff_down/pis_total[i][j]
                else:
                    diff_down = (scores_list[j]-scores_list[j-1])/2
                    diff_up = (scores_list[j+1]-scores_list[j])/2
                    step = (diff_down+diff_up)/pis_total[i][j]

            for k in range(0,int(pis_total[i][j])):
                if j == 0:
                    ref_applicants[i][pointer] = np.round(scores_list[j] + k*step) 
                else:
                    ref_applicants[i][pointer] = np.round(scores_list[j]-diff_down + k*step)
                pointer += 1
    
    return ref_applicants


# recalculate score for different banks
#applicants.shape = XxYxZ; X=Groups(white,black), Y=Banks, Z=Individual scores
def get_applicants(ref_applicants, score_shifts, score_range):
    applicants = []
    for i in range(0, len(ref_applicants)):
        applicants.append(np.zeros([len(score_shifts), len(ref_applicants[i])], dtype=np.int16))
        for j in range(0,len(applicants[i])):
            for k in range(0,len(applicants[i][j])):
                if ref_applicants[i][k] + score_shifts[j] < score_range[0]:
                    applicants[i][j][k] = score_range[0]
                elif ref_applicants[i][k] + score_shifts[j] > score_range[1]:
                    applicants[i][j][k] = score_range[1]
                else:
                    applicants[i][j][k] = ref_applicants[i][k] + score_shifts[j]
    return applicants

#applicant scores to cdfs
#applicant_cdfs.shape = XxYxZ, X=Groups, Y=Banks, Z= CDF for score range
def get_applicant_cdfs(applicants, scores_list):
    applicant_cdfs = np.ones([len(applicants), len(applicants[0]), len(scores_list)])
    for i in range(0,len(applicants)):
        for j in range(0, len(applicants[i])):
            pointer = 0
            for k in range(0, len(scores_list)):
                if k == len(scores_list)-1:
                    upper_thres = scores_list[k]
                else:
                    upper_thres = scores_list[k]+(scores_list[k+1]-scores_list[k])/2

                for l in range(pointer,len(applicants[i][j])):
                    if applicants[i][j][l] <= upper_thres:
                        pointer += 1 
                    else:
                        applicant_cdfs[i][j][k]=pointer/len(applicants[i][j])
                        break
    return applicant_cdfs

# to populate multiple group distributions
def get_applicant_pis(applicant_cdfs):
    pis=[]
    for i in range(0,len(applicant_cdfs)):
        tmp_pis = np.zeros([len(applicant_cdfs[i]), len(applicant_cdfs[i][0])])
        for j in range(len(applicant_cdfs[i])):
            tmp_pis[j][0] = applicant_cdfs[i][j][0]
            for score in range(len(applicant_cdfs[i][j])-1):
                tmp_pis[j][score+1] = applicant_cdfs[i][j][score+1] - applicant_cdfs[i][j][score]
        pis.append(tmp_pis)
    return pis


#calculate combined scores based on best interest rate
def get_combined_scores(applicant_scores, score_interest_rates,N_banks):
    combined_scores =[]
    for i in range(0,len(applicant_scores)):
        tmp_combined_scores = [np.zeros(len(applicant_scores[i][0]), dtype=np.int16)]
        for j in range(len(applicant_scores[i][0])-1,-1,-1):
            bank = get_minimal_interest_bank(score_interest_rates, applicant_scores, N_banks, i, j)
            tmp_combined_scores[0][j] = applicant_scores[i][bank][j]
                
        combined_scores.append(np.sort(tmp_combined_scores))        
            
    
    return combined_scores

## Old version...
#calculate combined scores based on best interest rate
def get_combined_scores2(applicant_scores, score_interest_intersect):
    combined_scores =[]
    for i in range(0,len(applicant_scores)):
        pointer = 0
        bank = 0
        tmp_combined_scores = np.zeros(len(applicant_scores[i][bank]))
        for j in range(len(applicant_scores[i][bank])-1,-1,-1):
            if applicant_scores[i][bank][j] > score_interest_intersect[pointer]:
                tmp_combined_scores[j] = applicant_scores[i][bank][j]
            else:
                if pointer < 1:
                    pointer += 1
                if bank < len(applicant_scores[i])-1:
                    bank += 1
                tmp_combined_scores[j] = applicant_scores[i][bank][j]
                
        combined_scores.append(tmp_combined_scores)        
            
    
    return combined_scores



#distribution if we take into account that applicants take loan with lowest interest rate
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


#depricated
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
