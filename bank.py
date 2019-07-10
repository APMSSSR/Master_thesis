import numpy as np
import matplotlib.pyplot as plt

class Bank:
    def __init__(self, name, color, line_style, interest_rate_range, market, score_shift=0, utility_repaid=1, utility_default= -4):
        self.name = name
        self.color = color
        self.line_style = line_style
        self.interest_rate_range = interest_rate_range
        self.score_shift = score_shift
        self.utility_repaid = utility_repaid
        self.utility_default = utility_default
        self.score_interest_rates = {}
        self.set_i_rates_mapping(market)
        self.expected_group_utility_curve = {}
        self.group_selection_rate = {}
        self.real_group_utility_curve = {}
        self.market_share = {}
        self.N_loan_curves = {}
        self.total_utility_curves = {}
        
    def set_i_rates_mapping(self, market):
        x_axis = np.linspace(market.score_range[0],market.score_range[1],market.score_range[1]-market.score_range[0]+1, dtype=int)
        y_axis = np.interp(x_axis, market.score_range, self.interest_rate_range)   
        self.score_interest_rates = dict(zip(x_axis, np.around(y_axis,4)))
        return self.score_interest_rates
    
    def get_expected_customer_score(self, market, customer_score):
        expected_customer_score = 0
        if customer_score + self.score_shift < market.score_range[0]:
            expected_customer_score = market.score_range[0]
        elif customer_score + self.score_shift > market.score_range[1]:
            expected_customer_score = market.score_range[1]
        else:
            expected_customer_score = customer_score + self.score_shift
        return expected_customer_score
    
    def get_customer_evaluation_utility(self, expected_customer_score, customer_group):
        return self.utility_default*(1-customer_group.score_repay_prob[expected_customer_score]) + (self.utility_repaid+self.score_interest_rates[expected_customer_score])*customer_group.score_repay_prob[expected_customer_score]
    
    def get_customer_utility(self, interest_rate, outcome):
        utility = 0
        if outcome:
            utility = self.utility_repaid+interest_rate
        else:
            utility = self.utility_default
        return utility
    
    def set_expected_group_utility_curve(self, customer_group, expected_utility):
        self.expected_group_utility_curve[customer_group.name] = expected_utility
        return self.expected_group_utility_curve[customer_group.name]
    
    def change_interest_rate(self, interest_change, market):
        if self.interest_rate_range[1] + interest_change >= 0:
            self.interest_rate_range[0] += interest_change
            self.interest_rate_range[1] += interest_change
            self.set_i_rates_mapping(market)
        return self.interest_rate_range
    
    def change_interest_rate_range(self, score_interest_rates, market):
        self.score_interest_rates = score_interest_rates
        self.interest_rate_range = [score_interest_rates[market.score_range[0]],score_interest_rates[market.score_range[1]]]

        return self.interest_rate_range
    
    
        
    def plot_expected_group_utility_curve(self, customer_group):
        plt.plot(list(range(0, len(self.expected_group_utility_curve[customer_group.name]))), self.expected_group_utility_curve[customer_group.name], color='black',LineStyle=':', label="expected bank utility curve")
        plt.ylabel('Utility')
        plt.xlabel('Customers')
        plt.title('Expected utility curve of ' + self.name + ' bank for ' + customer_group.name + ' group')
        plt.grid('on')
        plt.legend(loc="lower left")
        #plt.show()
        return 1
    
    def set_selection_rate(self, selection_rates):
        for group_name, selection_rate in selection_rates.items():
            self.group_selection_rate[group_name] = selection_rate     
        return 1
    
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)