import numpy as np
import random
import matplotlib.pyplot as plt
from applicant import Applicant

class Applicant_group:
    def __init__(self, name, color, line_style, size, scores, loan_demand, error_rate, score_error, market, repays, ir_limit=np.inf):
        self.name = name
        self.color = color
        self.line_style = line_style
        self.size = size
        self.loan_demand = loan_demand
        self.score_repay_prob = self.get_repay_prob_mapping(market.score_range, repays)
        self.initial_mean_score = np.mean(scores)
        
        real_scores = self.set_real_scores(scores, error_rate, score_error, market)
        self.applicants = self.create_applicants(scores, real_scores, ir_limit)
        self.sort_by_score()
    
    #Creates applicant objects and initializes them
    def create_applicants(self, scores, real_scores, ir_limit):
        applicants = []
        for i in range (0, self.size):
            applicants.append(Applicant(self, scores[i], real_scores[i], ir_limit))
        return applicants
    
    def sort_by_score(self):
        self.applicants.sort(key=lambda x: x.score, reverse=True)
        return self.applicants
        
    def get_repay_prob_mapping(self, score_range, repay_prob):
        x_axis = np.linspace(score_range[0],score_range[1],score_range[1]-score_range[0]+1, dtype=int)
        y_axis = np.interp(x_axis, repay_prob.index, repay_prob[self.name])
        return dict(zip(x_axis, y_axis))
    
    
    #simulating that some members of the group have better score/repay prob than rated
    def set_real_scores(self, scores, error_rate, score_error, market):
        real_scores = scores.copy()
        better_applicants = np.sort(random.sample(range(0, self.size), int(self.size*error_rate)))
        for applicant in better_applicants:
            if real_scores[applicant] + score_error < market.score_range[0]:
                real_scores[applicant] = market.score_range[0]
            elif real_scores[applicant] + score_error > market.score_range[1]:
                real_scores[applicant] = market.score_range[1]
            else:
                real_scores[applicant] += score_error
        return real_scores
    
    def get_scores(self):
        return list(applicant.score for applicant in self.applicants)
        
    def get_mean_score_change(self):
        return np.mean(self.get_scores())-self.initial_mean_score
  

    def plot_histogram(self, market):
        plt.figure()
        plt.hist(self.get_scores(), range = market.score_range, label='Step ' + str(market.step))
        plt.ylabel("Occurence")
        plt.xlabel("Score")
        plt.ylim([0,self.size*0.75])
        plt.legend(loc="upper left")
        plt.show()
        return 1
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
