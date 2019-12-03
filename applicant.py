import random

class Applicant:
    def __init__(self, group, score, real_score, ir_limit):
        self.group = group
        self.score = score
        self.repay_prob = group.score_repay_prob[score]
        self.real_score = real_score
        self.real_repay_prob = group.score_repay_prob[real_score]
        self.ir_limit = ir_limit
    
    def select_score_change(self, market, outcome):
        if outcome:
            return market.repay_score
        else:
            return market.default_score
    
    #get simulated real repay outcome of real score probability(1=repaid, 0=default ) and change the score and real score accordingly
    def get_repay_outcome(self, market):
        random_number = random.random()
        outcome = 1
        if random_number < 1-self.real_repay_prob:
            outcome = 0
        
        score_change = self.select_score_change(market, outcome)
        self.change_score(market, outcome, score_change)
        self.change_real_score(market, outcome, score_change)
        
        return outcome
    
    def change_score(self, market, outcome, score_change):
        if self.score + score_change < market.score_range[0]:
            self.score = market.score_range[0]
        elif self.score + score_change > market.score_range[1]:
            self.score = market.score_range[1]
        else:
            self.score += score_change
        
        return self.score
    
    def change_real_score(self, market, outcome, score_change):
        if self.real_score + score_change < market.score_range[0]:
            self.real_score = market.score_range[0]
        elif self.real_score + score_change > market.score_range[1]:
            self.real_score = market.score_range[1]
        else:
            self.real_score += score_change
        
        return self.real_score
    
    #get simulated expected repay outcome of score probability without change to the score(1=repaid, 0=default )
    def get_expected_repay_outcome(self):
        random_number = random.random()
        outcome = 1
        #print(random_number)
        if random_number < 1-self.repay_prob:
            outcome = 0
        return outcome