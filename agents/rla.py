class RLAlgorithm:
    # Produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action |action|
    # in state |state| resulted in reward |reward| and a transition to state |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")