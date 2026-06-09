from request import Request, RequestStatus, RequestPD
import heapq
from collections import deque

class Engine():
    def __init__(
        self, 
        requests: list[Request]
    ):
        self.pending: list[tuple[float, str, Request]] = []
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []
        for r in requests:
            r.status = RequestStatus.PENDING
            heapq.heappush(self.pending, (r.arrival_time, r.req_id, r))

    def release_arrivals(self, now: float):
        while self.pending and self.pending[0][0] <= now:
            _, _, r = heapq.heappop(self.pending)
            r.status = RequestStatus.WAITING
            self.waiting.append(r)
        
    def next_arrival(self):
        return self.pending[0][0] if self.pending else None

    def next(self):
        return self.next_arrival()

    def advance_to(self, time: float):
        self.release_arrivals(time)
         

