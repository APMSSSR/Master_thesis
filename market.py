import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")

class Market:
    def __init__(self, policy, policy_color, score_range = [300,850], repay_score = 75, default_score = -150, max_interest_rate_range =[0.5, 0.5], min_interest_rate_range=[0.001,0.001], plane_range=[0, 1], plane_slice_step=0.01):
        plane_range[1] += plane_slice_step 
        self.score_range = score_range
        self.policy = policy
        self.policy_color = policy_color
        self.repay_score = repay_score
        self.default_score = default_score
        self.max_interest_rate_range = max_interest_rate_range
        self.min_interest_rate_range = min_interest_rate_range
        self.interest_rate_plane = self.set_interest_rate_plane(plane_range, plane_slice_step)
        self.step = 0
        
        self.max_irates = []
        self.min_irates = []
        #loans and util on the whole market
        self.loans = {}
        self.utility = {}
    
    #create interest rate plane    
    def set_interest_rate_plane(self, plane_range, plane_slice_step):
        interest_rate_plane = {}
        plane_slices = np.arange(plane_range[0], plane_range[1] , plane_slice_step)
        for pslice in plane_slices:
            interest_rate_range = np.array(self.max_interest_rate_range) - (np.array(self.max_interest_rate_range) - np.array(self.min_interest_rate_range))*pslice
            x_axis = np.linspace(self.score_range[0], self.score_range[1], self.score_range[1]-self.score_range[0]+1, dtype=int)
            y_axis = np.interp(x_axis, self.score_range, interest_rate_range)
            interest_rate_plane[str(pslice)] = dict(zip(x_axis, np.around(y_axis,5)))

        return interest_rate_plane
    
    def get_selection_rate(self, banks, groups):
        if self.policy == "Max. utility":
            return self.get_MU_selection_rate(banks, groups)
        elif self.policy == "Dem. parity":
            return self.get_DP_selection_rate(banks, groups)
        elif self.policy == "Equal opportunity":
            return self.get_EO_selection_rate(banks, groups)
        else:
            return None
        
    def get_MU_selection_rate(self, banks, groups):
        #Get expected bank utility and set the bank selection rate
        selection_rates = {}
        for bank in banks:
            selection_rates[bank.name] = {}
            utility_curve = {}
            for group in groups:
                utility = 0
                utility_curve[group.name] = []
                for k in range(0, group.size):
                    customer_score = bank.get_expected_customer_score(self, group.scores[k])
                    utility += bank.get_customer_evaluation_utility(customer_score, group)
                    utility_curve[group.name].append(utility)
                bank.set_expected_group_utility_curve(group, utility_curve[group.name])
                #bank.plot_expected_group_utility_curve(group)

                selection_rates[bank.name][group.name] = (len(utility_curve[group.name]) - np.argmax(list(reversed(utility_curve[group.name])))-1)/group.size
                #print('Selection rate of ' + bank.name + ' bank for ' + group.name + ' group: ' + str(selection_rates[bank.name][group.name]))
        
        return selection_rates
    
    
    def get_DP_selection_rate(self, banks, groups):
        #Get expected bank utility and set the bank selection rate
        selection_rates = {}
        group_sizes = []
        for bank in banks:
            selection_rates[bank.name] = {}
            utility_curve = {}
            for group in groups:
                utility = 0
                utility_curve[group.name] = []
                group_sizes.append(group.size)
                for k in range(0, group.size):
                    customer_score = bank.get_expected_customer_score(self, group.scores[k])
                    utility += bank.get_customer_evaluation_utility(customer_score, group)
                    utility_curve[group.name].append(utility)
                
            merged_utility_curve = []
            for group in groups:
                if group.size is not max(group_sizes):             
                    x = list(range(0, group.size))
                    x = np.array(x)*(max(group_sizes)/group.size)
                    y = utility_curve[group.name]
                    z = np.polyfit(x, y, 3)
                    p = np.poly1d(z)
                    merged_utility_curve += p(list(range(0,max(group_sizes))))
                else:
                    merged_utility_curve += utility_curve[group.name]
            
            for group in groups:
                bank.set_expected_group_utility_curve(group, merged_utility_curve)
                #bank.plot_expected_group_utility_curve(group)
                selection_rates[bank.name][group.name] = (len(merged_utility_curve) - np.argmax(list(reversed(merged_utility_curve)))-1)/max(group_sizes)
                #print('Selection rate of ' + bank.name + ' bank for ' + group.name + ' group: ' + str(selection_rates[bank.name][group.name]))

        return selection_rates
    
    
    def get_EO_selection_rate(self, banks, groups):
        #Get expected bank utility and set the bank selection rate
        selection_rates = {}
        group_sizes = []
        for bank in banks:
            selection_rates[bank.name] = {}
            utility_curve = {}
            TPRs = {}
            for group in groups:
                utility = 0
                utility_curve[group.name] = []
                TPR = 0
                TPRs[group.name] = []
                group_sizes.append(group.size)
                
                for k in range(0, group.size):
                    customer_score = bank.get_expected_customer_score(self, group.scores[k])
                    utility += bank.get_customer_evaluation_utility(customer_score, group)
                    utility_curve[group.name].append(utility)
                    TPR = group.score_repay_prob[customer_score]
                    TPRs[group.name].append(TPR)
            
            #add TPR utility curves together
            main_group_name = groups[group_sizes.index(max(group_sizes))].name
            merged_TPR_utility_curve = utility_curve[main_group_name]
            for group in groups:
                if group.name is not main_group_name:
                    addition=0
                    pos=0
                    for k in range(len(merged_TPR_utility_curve)-1):
                        if TPRs[group.name][pos] <= TPRs[main_group_name][k] and TPRs[group.name][pos] >= TPRs[main_group_name][k+1]:
                            addition = utility_curve[group.name][pos]
                            pos += 1
                            if pos >= group.size:
                                merged_TPR_utility_curve[k] += addition
                                break
                        merged_TPR_utility_curve[k] += addition
                        
                bank.set_expected_group_utility_curve(group, utility_curve[group.name])
            
            #get TPR for max util
            selected_TPR = TPRs[main_group_name][(len(merged_TPR_utility_curve) - np.argmax(list(reversed(merged_TPR_utility_curve)))-1)]
            #print('Selected minimal TPR for ' + bank.name + ' bank is: ' + str(selected_TPR))
  
            #get selection rate
            for group in groups:
            
                    for k in range(group.size-1):
                        if TPRs[group.name][k+1] <= selected_TPR and TPRs[group.name][k] >= selected_TPR:
                            selection_rates[bank.name][group.name] = (k+1)/group.size
                            #print('Selection rate of ' + bank.name + ' bank for ' + group.name + ' group: ' + str(selection_rates[bank.name][group.name]))
                            break

        return selection_rates
        
    
    def plot_bank_interest_rates(self, banks):
        plt.figure(0)
        x_axis = self.score_range
        for bank in banks:
            y_axis = bank.interest_rate_range
            plt.plot(x_axis, y_axis ,color=bank.color ,LineStyle= bank.line_style, label= bank.name + " bank interest rates")
        plt.ylabel('Interest rate')
        plt.xlabel('Score')
        plt.title('Dependence of interest rate on score for different banks')
        plt.grid('on')
        plt.legend(loc="lower left")
        plt.show()
        return 1
    
    
    def plot_bank_utility_curves(self, banks, groups):
        fig, ax = plt.subplots(len(banks), len(groups),figsize=(16,8*len(banks))); 
        for i in range(len(banks)):
            for j in range(len(groups)):
                ax[i,j].plot(list(range(0,len(banks[i].real_group_utility_curve[groups[j].name]))), banks[i].real_group_utility_curve[groups[j].name], color='black',LineStyle=':', label="Bank utility curve")
                ax[i,j].set_title('Utility curve of ' + str(banks[i].name) + ' bank for ' + str(groups[j].name) + ' group')
                ax[i,j].set_xlabel('Customers')
                ax[i,j].set_ylabel('Bank utility')
                ax[i,j].legend(loc="upper left")
                ax[i,j].grid()
        return 1
    
    def plot_market_situation(self, banks, groups, mean_group_score_change_curve):
        fig, ax = plt.subplots(3,len(groups),figsize=(16,24))
        
        for i in range(len(groups)):
            ax[0][i].hist(groups[i].scores, range = self.score_range, label='Step ' + str(self.step))
            ax[0][i].set_title(groups[i].name + " group histogram change for " + self.policy + " policy")
            ax[0][i].set_ylabel("Occurence")
            ax[0][i].set_xlabel("Score")
            ax[0][i].set_ylim([0,groups[i].size*0.75])
            ax[0][i].grid()
            ax[0][i].legend(loc="upper left")
            
            y_axis = mean_group_score_change_curve[groups[i].name]
            ax[1][1].plot(list(range(len(mean_group_score_change_curve[groups[i].name]))),y_axis ,color=groups[i].color, label= groups[i].name + " group mean score")
                        
        ax[1][1].set_ylabel('Mean score')
        ax[1][1].set_xlabel('Step')
        ax[1][1].set_title('Mean score of different groups in time step:' + str(self.step))
        ax[1][1].grid()
        ax[1][1].legend(loc="lower left")
        
        
        x_axis = self.score_range
        for bank in banks:
            y_axis = bank.interest_rate_range
            ax[1][0].plot(x_axis, y_axis ,color=bank.color ,LineStyle= bank.line_style, label= bank.name + " bank interest rates")
            
            for group in groups:
                ax[2][0].plot(list(range(len(bank.N_loan_curves[group.name]))), bank.N_loan_curves[group.name], color=bank.color, LineStyle=group.line_style, label= bank.name + " bank: "+ group.name + " group")
                ax[2][1].plot(list(range(len(bank.total_utility_curves[group.name]))), bank.total_utility_curves[group.name], color=bank.color, LineStyle=group.line_style,label= bank.name + " bank: "+ group.name + " group")
            
        ax[1][0].set_ylabel('Interest rate')
        ax[1][0].set_xlabel('Score')
        ax[1][0].set_title('Dependence of interest rate on score for different banks in step:' + str(self.step))
        ax[1][0].grid()
        ax[1][0].legend(loc="lower left")
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[2][0].set_ylabel('Number of loans [log]')
        ax[2][0].set_title('Total number of loans given by bank to a group: step ' + str(self.step))
        ax[2][0].set_yscale('log')
        ax[2][0].grid()
        ax[2][0].legend()
        
        ax[2][1].set_ylabel('Utility [log]')
        ax[2][1].set_title('Total bank utility by group: step ' + str(self.step))
        ax[2][1].set_yscale('log')
        ax[2][1].grid()
        ax[2][1].legend()
        
        fig.savefig('../plots/MV3step'+ '%03d' % self.step +'.png')
                          
        return 1
    
    def plot_market_situation_PC(self, banks, groups, mean_group_score_change_curve):
        fig, ax = plt.subplots(3,len(groups),figsize=(16,20))
        
        for i in range(len(groups)):
            ax[0][i].hist(groups[i].scores, range = self.score_range, label='Step ' + str(self.step))
            ax[0][i].set_title(groups[i].name + " group histogram change for " + self.policy + " policy")
            ax[0][i].set_ylabel("Occurence")
            ax[0][i].set_xlabel("Score")
            ax[0][i].set_ylim([0,groups[i].size*0.75])
            ax[0][i].legend(loc="upper left")
            
            y_axis = mean_group_score_change_curve[groups[i].name]
            ax[1][1].plot(list(range(len(mean_group_score_change_curve[groups[i].name]))),y_axis ,color=groups[i].color, label= groups[i].name + " group mean score change")
                        
        ax[1][1].set_ylabel('Mean score change')
        ax[1][1].set_xlabel('Step')
        ax[1][1].set_title('Mean score change of different groups in time step:' + str(self.step))
        ax[1][1].grid()
        ax[1][1].legend(loc="lower left")
                     
        ax[1][0].plot(list(range(len(self.max_irates))), self.max_irates, color="red", label= "Max market interest rate for score 300")
        ax[1][0].plot(list(range(len(self.min_irates))), self.min_irates, color="green", label= "Min market interest rate for score 850")
        ax[1][0].set_ylabel('Interest rate')
        ax[1][0].set_xlabel('Step')
        ax[1][0].set_title('Interest rate step:' + str(self.step))
        ax[1][0].set_ylim([self.min_interest_rate_range[1],self.max_interest_rate_range[0]])
        ax[1][0].grid()
        ax[1][0].legend(loc="lower left")
        
        total_loans = np.zeros(self.step+1)
        total_utility = np.zeros(self.step+1)
        for group in groups:
            total_loans += np.array(self.loans[group.name])
            total_utility += np.array(self.utility[group.name])
            ax[2][0].plot(list(range(len(self.loans[group.name]))), self.loans[group.name], color = group.color, LineStyle = group.line_style, label = "Total loans "+ group.name + " group")
            ax[2][1].plot(list(range(len(self.utility[group.name]))), self.utility[group.name], color = group.color, LineStyle = group.line_style,label = "Total utility for "+ group.name + " group")
        
        ax[2][0].plot(list(range(len(total_loans))), total_loans, color="red", label= "Total loans")
        ax[2][0].set_ylabel('Number of loans')
        ax[2][0].set_xlabel('Step')
        ax[2][0].set_title('Number of loans given by banks to groups: step ' + str(self.step))
        #ax[2][0].set_yscale('log')
        ax[2][0].grid()
        ax[2][0].legend()
        
        ax[2][1].plot(list(range(len(total_utility))), total_utility, color="red",label= "Total utility")
        ax[2][1].set_ylabel('Utility')
        ax[2][1].set_xlabel('Step')
        ax[2][1].set_title('Bank utility by groups: step ' + str(self.step))
        #ax[2][1].set_yscale('log')
        ax[2][1].grid()
        ax[2][1].legend()
          
        fig.savefig('../plots/MV3step'+ '%03d' % self.step +'.png')
                          
        return 1
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)