"""
time_manager.py - This is my bot's time manager based on the 60 seconds per side per game we get.

basically this file makes sure my bot never goes over time during the game and also
limits how much time it spends on any given move to ensure consistant quality across game stages

Author - Graeme Huntley
"""

from typing import Optional


class TimeManager:
    """This class functions via a total time budget, expected move count, and
    and a set amout of reserve time for the expected move count. """
    
    def __init__(
        self, 
        total_budget: float = 60.0,
        expected_moves: int = 40,
        min_buffer_moves: int = 10
    ):
        """Used to initalize my time manager"""
        self.total_budget = total_budget
        self.expected_moves = expected_moves
        self.min_buffer_moves = min_buffer_moves
        self.time_used = 0.0
        self.move_times = []
        self.emergency_threshold = 8.0
        self.emergency_time_per_move = 0.15
        self.max_time_per_move = total_budget * 0.15
    
    def calculate_budget(self, move_count: int) -> float:
        """
        Calculates the time budget for the current move by doing the following:
        - checks to see how much time is left
        - estimates the likely number of moves remaining, always keeps enough time for 10 final moves to be safe
        - uses a game phase multiplier, moves 1-15 only get a 0.3-0.5 multiplyer, 16-30 get 0.5 to 1.2 and endgame gets 0.9
        - uses constraints to never go over time budget
        """
        time_remaining = self.total_budget - self.time_used
        if time_remaining < 1.0:
            return 0.05
        moves_remaining = max(
            self.expected_moves - move_count, 
            self.min_buffer_moves
        )
        
        base_time = time_remaining / moves_remaining
        phase_multiplier = self._get_phase_multiplier(move_count)
        time_budget = base_time * phase_multiplier
        
        if time_remaining < self.emergency_threshold:
            time_budget = min(time_budget, self.emergency_time_per_move)
        
        time_budget = self._apply_constraints(time_budget, time_remaining)
        
        return time_budget
    
    def _get_phase_multiplier(self, move_count: int) -> float:
        """As mentioned above this uses the current game phase to influance
        the ammount of time each move gets to have. Early game gets the least amount of time,
        mid game gets generally the most, end game gets the second most.
        """
        if move_count <= 15:
            return 0.3 + (move_count / 15.0) * 0.2
        elif move_count <= 30:
            progress = (move_count - 15) / 15.0
            return 0.5 + (progress * 0.7)
        elif move_count <= 50:
            progress = (move_count - 30) / 20.0
            return 1.2 - (progress * 0.3)
        else:
            return 0.9
    
    def _apply_constraints(
        self, 
        time_budget: float, 
        time_remaining: float
    ) -> float:
        """
        This function adds the constraints to ensure my 
        bot never goes over time during a game.
        """
        max_allowed = time_remaining * 0.80
        time_budget = min(time_budget, max_allowed)        
        time_budget = min(time_budget, self.max_time_per_move)        
        min_time = 0.1
        time_budget = max(time_budget, min_time)
        time_budget = min(time_budget, time_remaining - 0.5)  # Leave 0.5s buffer
        
        return time_budget
    
    def record_time(self, elapsed: float) -> None:
        """ As the name implies this is used to record and update the duration of a move and the total time used."""
        self.time_used += elapsed
        self.move_times.append(elapsed)
        moves_played = len(self.move_times)
        remaining = self.get_time_remaining()
        print(f"[Move {moves_played}] Used: {elapsed:.2f}s | Total: {self.time_used:.2f}s | Remaining: {remaining:.2f}s")
    
    def get_time_remaining(self) -> float:
        """Gets how much time the bot has left"""
        return max(0.0, self.total_budget - self.time_used)
    
    def get_average_time_per_move(self) -> float:
        """Does the math to see how much time left it has"""
        if not self.move_times:
            return 0.0
        return sum(self.move_times) / len(self.move_times)
    
    def is_emergency_mode(self) -> bool:
        """Checks the time to see if my bot is in emergancy speed mode"""
        return self.get_time_remaining() < self.emergency_threshold
    
    def get_efficiency_ratio(self) -> float:
        """Simply sees the average of how long moves take"""
        if not self.move_times:
            return 0.0
    
        moves_made = len(self.move_times)
        expected_time_per_move = self.total_budget / self.expected_moves
        expected_time_used = moves_made * expected_time_per_move
        
        if expected_time_used == 0:
            return 0.0
        
        return self.time_used / expected_time_used
    
    def reset(self) -> None:
        """Resets the amount of time available for new games"""
        self.time_used = 0.0
        self.move_times = []
    
    def get_stats(self) -> dict:
        """Gets time related stats for debugging"""
        return {
            'total_budget': self.total_budget,
            'time_used': self.time_used,
            'time_remaining': self.get_time_remaining(),
            'moves_played': len(self.move_times),
            'avg_time_per_move': self.get_average_time_per_move(),
            'efficiency_ratio': self.get_efficiency_ratio(),
            'emergency_mode': self.is_emergency_mode(),
            'time_pressure': self.time_used / self.total_budget,
        }
    
    def print_stats(self) -> None:
        """Prints nicely formatted stats"""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("TIME MANAGEMENT STATISTICS")
        print("="*50)
        print(f"Total budget:        {stats['total_budget']:.2f}s")
        print(f"Time used:           {stats['time_used']:.2f}s ({stats['time_pressure']*100:.1f}%)")
        print(f"Time remaining:      {stats['time_remaining']:.2f}s")
        print(f"Moves played:        {stats['moves_played']}")
        print(f"Avg per move:        {stats['avg_time_per_move']:.2f}s")
        print(f"Efficiency ratio:    {stats['efficiency_ratio']:.2f}")
        print(f"Emergency mode:      {'YES' if stats['emergency_mode'] else 'NO'}")
        print("="*50 + "\n")
    
    def __repr__(self) -> str:
        """Handy method for debugging with pretty text :D"""
        return (
            f"TimeManager(budget={self.total_budget:.1f}s, "
            f"used={self.time_used:.1f}s, "
            f"remaining={self.get_time_remaining():.1f}s, "
            f"moves={len(self.move_times)})"
        )
