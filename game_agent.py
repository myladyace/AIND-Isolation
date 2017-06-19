"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    The evaluation function is the # of legal moves of player minus the
    # of its opponent legal moves multiplied by two. When it is 0, the one whose
    current position is closer to the center(using manhattan distance) would 
    have more advantages.
    
    So if two players have the same # of available moves and the same distance
    from the center, return 0
    If current player has more(less) available moves or closer(further) to the
    center when both of them are at the same distance from the center, 
    return positive(negative). 
    If the player has won(lost) the game, return inf(-inf).
    

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    #Return inf(-inf) if the player have won(lost)
    if game.is_winner(player):
        return float('inf')
    elif game.is_loser(player):
        return float('-inf')
    
    #Calculate the number of available moves for player and its opponent
    player_legal_moves = len(game.get_legal_moves(player))
    opponent_legal_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    #Evaluation function is the difference between two players
    evaluation = float(player_legal_moves - 2*opponent_legal_moves)
    
    #Return the evaluation value if it is not 0
    if evaluation:
        return evaluation
    
    #If evaluation==0, the one who is closer to the center would have advantanges
    else:
        player_y, player_x = game.get_player_location(player)
        opponent_y, opponent_x = game.get_player_location(game.get_opponent(player))
        center_y, center_x = int(game.height//2), int(game.width//2)
        player_distance = abs(player_y - center_y) + abs(player_x - center_x)
        opponent_distance = abs(opponent_y - center_y) + abs(opponent_x - center_x)
        
        #Scale it by dividing the difference by the board height(width)
        #So the position effect should not be as critical as the diffrences of legal moves
        return float(player_distance-opponent_distance)/game.height
    
    
    raise NotImplementedError


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Here the evaluation function is simply the difference in # of legal moves 
    between two players.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Return inf(-inf) if the player have won(lost)
    if game.is_winner(player):
        return float('inf')
    elif game.is_loser(player):
        return float('-inf')
    
    #Calculate the number of available moves for player and its opponent
    player_legal_moves = len(game.get_legal_moves(player))
    opponent_legal_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    #Return difference of legal moves between two players
    return float(player_legal_moves - opponent_legal_moves)
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    
    Here the heuristic is more aggressive while we try to block the opponent
    more. The evaluation function is the # of legal moves of player minus the
    # of its opponent legal moves multiplied by two.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #Return inf(-inf) if the player have won(lost)
    if game.is_winner(player):
        return float('inf')
    elif game.is_loser(player):
        return float('-inf')
    
    #Calculate the number of available moves for player and its opponent
    player_legal_moves = len(game.get_legal_moves(player))
    opponent_legal_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    #Return difference of legal moves between two players
    return float(player_legal_moves - 2*opponent_legal_moves)
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move if timeout
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        #Initialize the move, if there is no available move, will return (-1,-1)
        output_move = (-1,-1)
        
        #Initialize the evaluation score
        output_score = float('-inf')
        
        #Get possible moves for current player
        possible_moves = game.get_legal_moves()
        
        #Current layer is a maximizing layer
        for move in possible_moves:
            #Evaluate this move by calling helper function
            score = self.helper(game.forecast_move(move),depth-1)
            
            #Already find a winning move
            if score == float('inf'):
                return move
            
            #Update the score and move if current move is best so far
            if score>output_score:
                output_score = score
                output_move = move
                   
        return output_move

        raise NotImplementedError
    
    def helper(self, game, depth, maximize = False):
        """Evaluate the score of current move
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            
        maximize : bool
            This value indicates whether the current search layer is a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score of current move

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth==0:
            return self.score(game, self)

        #Initialize the evaluation score
        output_score = float('-inf') if maximize else float('inf')
        
        #Get possible moves for current player
        possible_moves = game.get_legal_moves()
        
        #if current layer is maximizing layer:
        if maximize:
            for move in possible_moves:
                #Evaluate this move by calling helper function
                score = self.helper(game.forecast_move(move),depth-1,not maximize)
            
                #Already find a winning move
                if score == float('inf'):
                    return score
            
                #Update the score and move if current move is best so far
                if score>output_score:
                    output_score=score
        
        #Minimizing Layer
        else:
            for move in possible_moves:
                score = self.helper(game.forecast_move(move),depth-1,not maximize)
                if score == float('-inf'):
                    return score
                if score<output_score:
                    output_score=score
        
        return output_score

        raise NotImplementedError


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            max_depth = len(game.get_blank_spaces())
            for depth in range(1,max_depth):
                best_move = self.alphabeta(game, depth)
            return best_move
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
        raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximize=True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers
            
        maximize : bool
            This value indicates whether the current search layer is a
            maximizing layer (True) or a minimizing layer (False)


        Returns
        -------
        float
            The score of current move
            
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth==0:
            return self.score(game, self)
        
        #Initialize the move, if there is no available move, will return (-1,-1)
        output_move = (-1,-1)

        #Initialize the evaluation score
        output_score = float('-inf') if maximize else float('inf')
        
        #Get possible moves for current player
        possible_moves = game.get_legal_moves()
        
        for move in possible_moves:
            #Evaluate this move by calling helper function
            score = self.helper(game.forecast_move(move),depth-1,alpha, beta)
            
            #In a maximizing layer, if the score of next move is greater than upper bound
            #we can simply cut this branch
            if score >= beta:
                return move
                
            #Update the score and move if current move is best so far
            if score>output_score:
                output_move = move
                output_score = score
                
            #Update the alpha value
            alpha = max(alpha, output_score)
       
        return output_move

        raise NotImplementedError
        
        
    def helper(self, game, depth, alpha, beta, maximize = False):
        """Evaluate the score of current move
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            
        maximize : bool
            This value indicates whether the current search layer is a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score of current move

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        if depth==0:
            return self.score(game, self)


        #Initialize the evaluation score
        output_score = float('-inf') if maximize else float('inf')
        
        #Get possible moves for current player
        possible_moves = game.get_legal_moves()
        
        #if current layer is maximizing layer:
        if maximize:
            for move in possible_moves:
                #Evaluate this move by calling helper function
                score = self.helper(game.forecast_move(move),depth-1,alpha, beta, not maximize)
            
                #In a maximizing layer, if the score of next move is greater than upper bound
                #we can simply cut this branch
                if score >= beta:
                    return score
                
                #Update the score and move if current move is best so far
                if score>output_score:
                    output_score = score
                
                #Update the alpha value
                alpha = max(alpha, output_score)
        
        #Minimizing Layer
        else:
            for move in possible_moves:
                score = self.helper(game.forecast_move(move),depth-1,alpha, beta, not maximize)
                if score <= alpha:
                    return score
                if score<output_score:
                    output_score = score
                beta = min(beta, output_score)
        
        return output_score

        raise NotImplementedError
    
    
