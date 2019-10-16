from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
import game


import sys
sys.path.append("./teams/HungryPacman/")

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveAgent', second='DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def getGameState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()


class PositionSearchProblem(SearchProblem):

    def __init__(self, gameState, costFn=lambda x: 1, goalStates=None, startState=None, visualize=False):
        self.gameState = gameState
        self.walls = gameState.getWalls()
        self.startState = startState
        self.goalStates = goalStates
        self.costFn = costFn

        # For display purposes
        self.visualize = visualize
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def getGameState(self):
        return self.gameState

    def isGoalState(self, state):
        # For display purposes only
        if state in self.goalStates and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return state in self.goalStates

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = game.Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
    Returns the cost of a particular sequence of actions. If those actions
    include an illegal move, return 999999.
    """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = game.Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


def astarSearch(searchProblem, heuristic):
    startState = searchProblem.getStartState()
    queue = util.PriorityQueue()
    closed = set()
    queue.push((startState, [], 0), heuristic(startState, searchProblem))
    # print(heuristic(startState, searchProblem))
    while not queue.isEmpty():
        state, actions, cost = queue.pop()
        closed.add(state)
        if searchProblem.isGoalState(state):
            return actions
        successors = searchProblem.getSuccessors(state)
        for s, a, c in successors:
            if s not in closed:
                na = list(actions)
                na.append(a)
                c = cost + c
                queue.push((s, na, c), c + heuristic(s, searchProblem))
    return []


def nullHeuristic(state, problem=None):
    return 0


#########################
# Classic Search Agents #
#########################

class BasicAgent(CaptureAgent):
    """
    Initialization section
    """

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        
        """
        OFFENSIVE PART 1
        """
        self.startGameState = gameState
        self.homeBoundary = self.getHomeList(gameState)
        self.gridMap = {}
        self.gridSafeMap = {}
        self.mapGrid(gameState)
        myTeam = self.getTeam(gameState)
        if self.index == myTeam[0]:
            self.mate = myTeam[1]
        else:
            self.mate = myTeam[0]
        self.enemyPred = [None, None]
        # self.enemyPred = [[None, None], [None, None]]

        # print("---- grid map ----")
        # print(self.gridSafeMap)
        self.debugDrawDFS()

        self.foodFactor = pow(self.getFoodNum(gameState), 0.5)
        
        
        """
        DEFENSIVE PART 1
        """
        self.teammate = self.getTeam(gameState)[1 - self.getTeam(gameState).index(self.index)]
        self.start = gameState.getAgentPosition(self.index)
        self.food = self.getFood(gameState)
        self.foodNum = self.getFoodNum(gameState)
        self.home = self.getHome(gameState)
        self.scoreHistory = [0]
        self.enemyHistory = [[((gameState.data.layout.width - self.start[0] - 1, gameState.data.layout.height - self.start[1] - 1), 0, 1), ((gameState.data.layout.width - gameState.getAgentPosition(self.teammate)[0] - 1, gameState.data.layout.height - gameState.getAgentPosition(self.teammate)[1] - 1), 0, 1)]]
        self.choice = [True, True]

        
    
    """
    OFFENSIVE PART 2
    """
    """
    Grid mapping
    """

    def mapGrid(self, gameState):
        startState = gameState.getAgentState(self.index).getPosition()
        self.dfsMap(gameState, startState)
        self.defineDeadState(gameState)
        # print("---- grid map ----\n", self.gridMap)
        self.defineLiveState(gameState)
        self.defineSafeDist(gameState)

    def dfsMap(self, gameState, startState):
        searchProblem = PositionSearchProblem(gameState=gameState, startState=startState)
        startState = searchProblem.getStartState()
        stack = util.Stack()
        stack.push((startState, []))
        closed = set()
        while not stack.isEmpty():
            state, path = stack.pop()
            if state not in closed:
                closed.add(state)
                successors = searchProblem.getSuccessors(state)
                self.mapBySuccessor(gameState, state, successors)
                for s, a, c in successors:
                    p = list(path)
                    p.append(state)
                    if s in closed and state != startState:
                        if p[-2] != s:
                            self.mapByVisited(s, p)
                    else:
                        stack.push((s, p))

    def mapBySuccessor(self, gameState, state, successors):
        successorNum = len(successors)
        if state not in self.gridMap:
            if successorNum == 1:
                self.gridMap[state] = "dead"
            elif successorNum == 2:
                # v1 = Actions.directionToVector(successors[0][1])
                # v2 = Actions.directionToVector(successors[1][1])
                # mult = v1[0] * v2[0] + v1[1] * v2[1]
                # sum = [state[0] + v1[0] + v2[0], state[1] + v1[1] + v2[1]]
                # if mult == 0 and not gameState.hasWall(int(sum[0]), int(sum[1])):
                #   self.gridMap[state] = "corner"
                #   # print(state)
                # else:
                #   self.gridMap[state] = "linked"
                self.gridMap[state] = "linked"
            elif successorNum >= 3:
                self.gridMap[state] = "cross"

    def mapByVisited(self, state, path):
        path.reverse()
        for s in path:
            if s == state: break
            # if self.gridMap[s] == "corner": continue
            self.gridMap[s] = "live_like"

    def defineDeadState(self, gameState):
        dead = [state for state in self.gridMap if self.gridMap[state] == "dead"]
        for d in dead:
            searchProblem = PositionSearchProblem(gameState=gameState)
            stack = util.Stack()
            stack.push(d)
            closed = set()
            while not stack.isEmpty():
                state = stack.pop()
                if state not in closed:
                    closed.add(state)
                    successors = searchProblem.getSuccessors(state)
                    for s, a, c in successors:
                        if self.gridMap[s] != "live_like":
                            self.gridMap[s] = "dead"
                            stack.push(s)
        # corner = [state for state in self.gridMap if self.gridMap[state] == "corner"]
        # for c in corner:
        #   self.gridMap[c] = "dead"

    def defineLiveState(self, gameState):
        live_like = [state for state in self.gridMap if self.gridMap[state] == "live_like"]
        live = set(self.getHomeList(gameState))
        for d in live_like:
            searchProblem = PositionSearchProblem(gameState=gameState, goalStates=live)
            stack = util.Stack()
            stack.push(d)
            closed = set()
            visited = []
            is_dead = True
            while not stack.isEmpty():
                state = stack.pop()
                closed.add(state)
                visited.append(state)
                if searchProblem.isGoalState(state):
                    is_dead = False
                    break
                successors = [s for s, a, c in searchProblem.getSuccessors(state) if self.gridMap[s] != "dead"]
                # print(state, successors)
                for s in successors:
                    if s not in closed:
                        stack.push(s)
            for s in visited:
                if is_dead:
                    self.gridMap[s] = "dead"
                else:
                    self.gridMap[s] = "live"
                    live.add(s)

    def defineSafeDist(self, gameState):
        goal = [state for state in self.gridMap if self.gridMap[state] == 'live']
        for state in self.gridMap:
            searchProblem = PositionSearchProblem(gameState=gameState, goalStates=goal, startState=state)
            actions = astarSearch(searchProblem, nullHeuristic)
            self.gridSafeMap[state] = len(actions)

    def debugDrawDFS(self):
        dead = []
        live = []
        for state in self.gridSafeMap:
            if self.gridSafeMap[state] == 0:
                live.append(state)
            else:
                dead.append(state)

        self.debugDraw(dead, [100, 125, 150], False)
        self.debugDraw(live, [125, 150, 100], False)

    """
    Action Section
    """

    def chooseAction(self, gameState):
        # actions = gameState.getLegalActions(self.index)
        # return random.choice(actions)
        self.updateEnemyPred(gameState)

        carryNum = self.getCarryingNum(gameState)
        homeList = self.getHomeList(gameState)
        capsuleList = self.getCapsuleList(gameState)
        foodList = self.getFoodList(gameState)
        enemyList = self.getEnemy(gameState)

        costFn = self.simpleCostFn
        heuristic = lambda x, y: self.simpleFoodSearchHeuristic(x, y, foodList, homeList, carryNum, enemyList,
                                                                capsuleList)
        goal = self.foodGoal(gameState)

        if not self.isAtHome(self.getCurrentPosition(gameState), gameState) and self.getCarryingNum(gameState) != 0:
            goal = self.foodHomeGoal(gameState)

        searchProblem = PositionSearchProblem(gameState=gameState,
                                              costFn=costFn,
                                              goalStates=goal,
                                              startState=self.getCurrentPosition(gameState))
        # self.debugDraw(goal, [100, 100, 100], False)
        actions = astarSearch(searchProblem, heuristic)
        return actions[0]

    def updateEnemyPred(self, gameState):
        enemy = self.getEnemy(gameState)
        for i in range(2):
            pos = enemy[i].getPosition()
            if pos is not None:
                self.enemyPred[i] = enemy[i]
            #   self.enemyPred[i] = [enemy[i], 0]
            # elif self.enemyPred[i][0] is not None:
            #   self.enemyPred[i][1] -= 1

    """
    Search Goals
    """

    def foodHomeGoal(self, gameState):
        goalStates = []
        goalStates.extend(self.getFoodList(gameState))
        goalStates.extend(self.getHomeList(gameState))
        goalStates.extend(self.getCapsuleList(gameState))
        return goalStates

    def foodGoal(self, gameState):
        goalStates = []
        goalStates.extend(self.getFoodList(gameState))
        goalStates.extend(self.getCapsuleList(gameState))
        return goalStates

    def homeGoal(self, gameState):
        return self.getHomeList(gameState)

    def defendTargetGoal(self, gameState):
        goalStates = [enemy.getPosition() for enemy in self.getEnemy(gameState) if enemy.getPosition()]
        return goalStates

    """
    Heuristic
    """

    def heuristicWithMate(self, heuristic, gameState, state):
        pos = gameState.getAgentState(self.mate).getPosition()
        dist = self.getMazeDistance(pos, state)
        return heuristic * (1 + 3 / (dist + 2))
        # if dist < 2:
        #   return heuristic * 2
        # else:
        #   return heuristic

    def simpleFoodSearchHeuristic(self, state, problem, foodList, homeList, carryNum, enemyList, capsuleList):
        gameState = problem.getGameState()
        heuristic = self.getFoodWeight(state, foodList)
        heuristic += self.getHomeWeight(state, homeList, carryNum)
        heuristic += self.getEnemyWeight(state, enemyList)
        heuristic += self.getCapsuleWeight(state, capsuleList)
        return self.heuristicWithMate(heuristic, gameState, state)

    def getEnemyWeight(self, state, enemyList):
        enemyDistance = self.getEnemyDistance(state, enemyList)
        if enemyDistance:
            dist = min(enemyDistance)
            if dist <= 1:
                return 999999
            else:
                return 10 / (dist - 1)
        else:
            return 0

    def getFoodWeight(self, state, foodList):
        food_dist = self.getFoodDistance(state, foodList)
        if food_dist:
            return min(food_dist)
        else:
            return 0

    def getHomeWeight(self, state, homeList, carryNum):
        dist = self.getHomeDistance(state, homeList)
        return dist * pow(carryNum, 0.5) / self.foodFactor

    def getCapsuleWeight(self, state, capsuleList):
        capsule = self.getCapsuleDistance(state, capsuleList)
        if capsule:
            return min(capsule) / 2
        else:
            return 0

    def simpleFoodSearchCost(self, start, goal, gamaState):
        problem = PositionSearchProblem(gameState=gamaState, costFn=self.simpleCostFn,
                                        goalStates=goal, startState=start)
        return problem.getCostOfActions(astarSearch(problem, nullHeuristic))

    def simpleCostFn(self, state):
        if self.gridMap[state] == "live":
            return 1
        elif self.gridMap[state] == "dead":
            return 2

    """
    help methods
    """

    def isScared(self, gameState):
        myState = gameState.getAgentState(self.index)
        return myState.scaredTimer != 0

    def getEnemy(self, gameState):
        enemyList = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        return enemyList

    def getEnemyDistance(self, state, enemyList):
        enemyDistance = [self.getMazeDistance(state, enemy.getPosition()) for enemy in
                         enemyList if not enemy.isPacman and enemy.getPosition() is not None and enemy.scaredTimer <= 2]
        return enemyDistance

    # def getEnemyPredDistance(self, state, enemyList):
    #   enemyDistance = [self.getMazeDistance(state, enemy[0].getPosition()) - enemy[1] for enemy in
    #                    enemyList if not enemy[0] is None if not enemy[0].isPacman
    #                    and enemy[0].getPosition() is not None and enemy[0].scaredTimer <= 2]
    #   return enemyDistance

    def getCurrentPosition(self, gameState):
        return gameState.getAgentState(self.index).getPosition()

    def getFoodDistance(self, state, foodList):
        foodDistance = [self.getMazeDistance(state, food) for food in foodList]
        return foodDistance

    def getFoodList(self, gameState):
        return self.getFood(gameState).asList()

    def getFoodNum(self, gameState):
        return len(self.getFoodList(gameState))

    def getCapsuleDistance(self, state, capsuleList):
        capsuleDistance = [self.getMazeDistance(state, capsule) for capsule in capsuleList]
        return capsuleDistance

    def getCapsuleList(self, gameState):
        return self.getCapsules(gameState)

    def getHomeDistance(self, state, homeList):
        homeDistance = [self.getMazeDistance(state, home) for home in homeList]
        return min(homeDistance)

    def getHomeList(self, gameState):
        homeList = []
        if self.red:
            x = gameState.data.layout.width // 2 - 1
        else:
            x = gameState.data.layout.width // 2
        for y in range(1, gameState.data.layout.height):
            if not gameState.hasWall(x, y):
                homeList.append((x, y))
        return homeList

    def getCarryingNum(self, gameState):
        return gameState.getAgentState(self.index).numCarrying

    def isAtHome(self, state, gameState):
        if self.red:
            return state[0] < gameState.data.layout.width // 2
        else:
            return state[0] >= gameState.data.layout.width // 2

    def isAtOpponent(self, state, gameState):
        return not self.isAtHome(state, gameState)
    
    
    """
    OFFENSIVE PART 2
    """

    def deadList(self):
        return [state for state in self.gridMap if self.gridMap[state] == 'dead']

    def liveList(self):
        return [state for state in self.gridMap if self.gridMap[state] == 'live']


    def teamIsPanman(self, gameState, state):
        if self.red:
            return state[0] > gameState.data.layout.width // 2 - 1
        else:
            return state[0] < gameState.data.layout.width // 2

    def opponentIsPacman(self, gameState, state):
        return not self.teamIsPanman(gameState, state)

    def nearestLive(self, gameState, state):
        problem = PacmanContestProblem(self, gameState, self.liveList(), start=state)
        state, actions = breadthFirstSearch(problem)
        return state

    def getHome(self, gameState):
        homeList = []
        if gameState:
            x = gameState.data.layout.width // 2 - 1
        else:
            x = gameState.data.layout.width // 2
        for y in range(1, gameState.data.layout.height):
            if not gameState.hasWall(x, y) and ((self.red and not gameState.hasWall(x + 1, y)) or (
                    not self.red and not gameState.hasWall(x - 1, y))):
                homeList.append((x, y))
        return homeList

    def getEnermy(self, gameState):
        return [gameState.getAgentState(i) for i in self.getOpponents(gameState) if
                gameState.getAgentState(i).getPosition() != None]

    def getScoreHistory(self, gameState):
        self.scoreHistory.append(self.getScore(gameState))

    def getEnermyHistory(self, gameState):
        self.getScoreHistory(gameState)
        curFood = self.getFoodYouAreDefending(gameState).asList()
        if self.getPreviousObservation():
            preFood = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
        else:
            preFood = curFood
        enemyObserved = [(enemy.getPosition(), enemy.numCarrying, 1) for enemy in self.getEnermy(gameState)]
        foodEaten = [food for food in preFood if food not in curFood]
        enemy1, enemy2 = self.enemyHistory[-1]
        if self.scoreHistory[-1] < self.scoreHistory[-2]:
            if self.scoreHistory[-2] - self.scoreHistory[-1] == enemy1[1]:
                enemy1 = (enemy1[0], 0, enemy1[2])
            elif self.scoreHistory[-2] - self.scoreHistory[-1] == enemy2[1]:
                enemy2 = (enemy2[0], 0, enemy2[2])
            elif self.scoreHistory[-2] - self.scoreHistory[-1] == enemy1[1] + enemy2[1]:
                enemy1 = (enemy1[0], 0, enemy1[2])
                enemy2 = (enemy2[0], 0, enemy2[2])
        if len(preFood) < len(curFood):
            if len(curFood) - len(preFood) + len(foodEaten) == enemy1[1] and enemy1[2] == 1:
                enemy1 = (
                (gameState.data.layout.width - self.start[0] - 1, gameState.data.layout.height - self.start[1] - 1), 0,
                1)
            elif len(curFood) - len(preFood) + len(foodEaten) == enemy2[1] and enemy2[2] == 1:
                enemy2 = (
                (gameState.data.layout.width - self.start[0] - 1, gameState.data.layout.height - self.start[1] - 1), 0,
                1)
            elif len(curFood) - len(preFood) + len(foodEaten) == enemy1[1] + enemy2[1] and enemy1[2] == 1 and enemy2[
                2] == 1:
                enemy1 = (
                (gameState.data.layout.width - self.start[0] - 1, gameState.data.layout.height - self.start[1] - 1), 0,
                1)
                enemy2 = (
                (gameState.data.layout.width - self.start[0] - 1, gameState.data.layout.height - self.start[1] - 1), 0,
                1)
        if len(enemyObserved) == 2:
            self.enemyHistory.append(enemyObserved)
        elif len(enemyObserved) == 1:
            if len(foodEaten) == 2:
                if enemyObserved[0][1] == enemy1[1] + 1:
                    foodNum = enemy2[1] + 1;
                elif enemyObserved[0][1] == enemy2[1] + 1:
                    foodNum = enemy1[1] + 1;
                else:
                    foodNum = enemy1[1] + enemy2[1] + 2 - enemyObserved[0][1]
                if enemyObserved[0][0] == foodEaten[0]:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[1], foodNum, 1)])
                else:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[0], foodNum, 1)])
            elif len(foodEaten) == 1 and enemyObserved[0][0] != foodEaten[0]:
                if enemyObserved[0][1] == enemy1[1] and self.getMazeDistance(enemyObserved[0][0], enemy1[0]) <= enemy1[
                    2] and self.getMazeDistance(foodEaten[0], enemy2[0]) <= enemy2[2]:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[0], enemy2[1] + 1, 1)])
                elif enemyObserved[0][1] == enemy2[1] and self.getMazeDistance(enemyObserved[0][0], enemy2[0]) <= \
                        enemy2[2] and self.getMazeDistance(foodEaten[0], enemy1[0]) <= enemy1[2]:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[0], enemy1[1] + 1, 1)])
                elif self.getMazeDistance(enemyObserved[0][0], enemy1[0]) <= enemy1[2] and self.getMazeDistance(
                        foodEaten[0], enemy2[0]) <= enemy2[2]:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[0], enemy2[1] + 1, 1)])
                else:
                    self.enemyHistory.append([enemyObserved[0], (foodEaten[0], enemy1[1] + 1, 1)])
            else:
                if enemyObserved[0][1] == enemy1[1] + len(foodEaten) and self.getMazeDistance(enemyObserved[0][0],
                                                                                              enemy1[0]) <= enemy1[2]:
                    self.enemyHistory.append([enemyObserved[0], (enemy2[0], enemy2[1], enemy2[2] + 1)])
                elif enemyObserved[0][1] == enemy2[1] + len(foodEaten) and self.getMazeDistance(enemyObserved[0][0],
                                                                                                enemy2[0]) <= enemy2[2]:
                    self.enemyHistory.append([enemyObserved[0], (enemy1[0], enemy1[1], enemy1[2] + 1)])
                elif self.getMazeDistance(enemyObserved[0][0], enemy1[0]) <= enemy1[2]:
                    self.enemyHistory.append([enemyObserved[0], (enemy2[0], enemy2[1], enemy2[2] + 1)])
                else:
                    self.enemyHistory.append([enemyObserved[0], (enemy1[0], enemy1[1], enemy1[2] + 1)])
        else:
            if len(foodEaten) == 2:
                self.enemyHistory.append([(enemy1[0], enemy1[1] + 1, 1), (enemy2[0], enemy2[1] + 1, 1)])
            elif len(foodEaten) == 1:
                if self.getMazeDistance(foodEaten[0], enemy1[0]) <= enemy1[2]:
                    self.enemyHistory.append([(foodEaten[0], enemy1[1] + 1, 1), (enemy2[0], enemy2[1], enemy2[2] + 1)])
                else:
                    self.enemyHistory.append([(foodEaten[0], enemy2[1] + 1, 1), (enemy1[0], enemy1[1], enemy1[2] + 1)])
            else:
                self.enemyHistory.append([(enemy1[0], enemy1[1], enemy1[2] + 1), (enemy2[0], enemy2[1], enemy2[2] + 1)])

"""
OFFENSIVE PART 3
"""

class OffensiveAgent(BasicAgent):
  def chooseAction(self, gameState):
    self.updateEnemyPred(gameState)
    carryNum = self.getCarryingNum(gameState)
    homeList = self.getHomeList(gameState)
    capsuleList = self.getCapsuleList(gameState)
    foodList = self.getFoodList(gameState)
    enemyList = [x for x in self.enemyPred if x is not None]

    pos = self.getCurrentPosition(gameState)
    costFn = lambda x: self.saerchCostFn(state=x)
    goal = self.foodGoal(gameState)
    heuristic = lambda x, y: self.simpleFoodSearchHeuristic(x, y, foodList, homeList, carryNum, enemyList, capsuleList)
    start = time.time()

    """
    Decision tree
    """
    if self.isAtHome(pos, gameState):
      dist = self.getEnemyDistance(pos, enemyList)
      if dist:
        if min(dist) > 5:
          enemyList = []
      if pos in self.getHomeList(gameState):
        pass
      else:
        pass
      heuristic = lambda x, y: self.simpleFoodSearchHeuristic(x, y, foodList, homeList, carryNum, enemyList,
                                                              capsuleList)
    else:
      dist = self.getEnemyDistance(pos, enemyList)
      if dist:
        min_dist = min(dist)
        # heuristic = self.simpleFoodSearchHeuristic
        costFn = lambda x: self.saerchCostFn(x, enemy_dist=min_dist)
        if min_dist > 2:
          if self.getCarryingNum(gameState) == 0:
            goal = self.foodGoal(gameState)
          else:
            goal = self.foodHomeGoal(gameState)
        else:
          goal = self.escapeGoal(gameState)
          costFn = self.simpleCostFn
      else:
        heuristic = lambda x, y: self.searchFoodHeuristic(x, y, foodList, homeList, carryNum, capsuleList)
        if self.getCarryingNum(gameState) == 0:
          goal = self.foodGoal(gameState)
        else:
          goal = self.foodHomeGoal(gameState)
      if len(self.getFoodList(gameState)) < 3:
        if self.getCarryingNum(gameState) == 0:
          goal = self.foodGoal(gameState)
        else:
          goal = self.homeGoal(gameState)

    searchProblem = PositionSearchProblem(gameState=gameState,
                                          costFn=costFn,
                                          goalStates=goal,
                                          startState=self.getCurrentPosition(gameState),
                                          visualize=False)
    # self.debugDraw(goal, [100, 100, 100], False)
    actions = astarSearch(searchProblem, heuristic)

    print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

    if len(actions) != 0:
      return actions[0]
    else:
      actions = gameState.getLegalActions(self.index)
      return random.choice(actions)

  def escapeGoal(self, gameState):
    goalStates = []
    goalStates.extend(self.getCapsuleList(gameState))
    goalStates.extend(self.getHomeList(gameState))
    return goalStates

  def searchFoodHeuristic(self, state, problem, foodList, homeList, carryNum, capsuleList):
    gameState = problem.getGameState()
    heuristic = self.getHomeWeight(state, homeList, carryNum)
    heuristic += self.getFoodWeight(state, foodList)
    heuristic += self.getCapsuleWeight(state, capsuleList) * 4
    return self.heuristicWithMate(heuristic, gameState, state)

  def saerchCostFn(self, state, enemy_dist=None):
    # if self.isAtHome(state, self.startGameState):
    #   return 1
    if enemy_dist is None:
      if self.gridSafeMap[state] < 4: # TODO scan map
        return 1
      else:
        return 2
    else:
      if self.gridSafeMap[state] <= (enemy_dist - 1) // 2:
        return 1
      else:
        return 999999


"""
DEFFENSIVE PART 3
"""

class DefensiveAgent(OffensiveAgent):
    def chooseAction(self, gameState):
        self.getEnermyHistory(gameState)

        self.debugClear()
        self.debugDraw([self.enemyHistory[-1][0][0], self.enemyHistory[-1][1][0]], [200, 200, 200])

        foodDepth = [self.gridSafeMap[food] for food in self.getFood(gameState).asList()]
        avg = sum(foodDepth) / len(foodDepth)
        enemyFoodDepth = [self.gridSafeMap[food] for food in self.getFoodYouAreDefending(gameState).asList()]
        enemyavg = sum(enemyFoodDepth) / len(enemyFoodDepth)
        if self.isAtHome(gameState.getAgentPosition(self.index), gameState):
            if self.getScore(gameState) > 0: #self.foodNum / 5) and enemyavg > 3:
                self.choice[self.getTeam(gameState).index(self.index)] = False
            else:
                self.choice = [True, True]
        if self.isAtHome(gameState.getAgentPosition(self.teammate), gameState):
            if self.getScore(gameState) > 0: #self.foodNum / 5 and enemyavg > 3:
                self.choice[self.getTeam(gameState).index(self.teammate)] = False
            else:
                self.choice = [True, True]
        if self.isScared(gameState):
            self.choice = [True, True]
        print(self.choice)

        if self.choice[self.getTeam(gameState).index(self.index)]:
            return super().chooseAction(gameState)

        self.getCapsules(gameState)
        enemy1, enemy2 = self.enemyHistory[-1]
        if enemy1[1] > enemy2[1] or (enemy1[1] == enemy2[1] and enemy1[2] <= enemy2[2]):
            if not self.isAtHome(enemy2[0], gameState):
                enemylist = [enemy1, enemy1]
            else:
                enemylist = [enemy1, enemy2]
        else:
            if not self.isAtHome(enemy1[0], gameState):
                enemylist = [enemy2, enemy2]
            else:
                enemylist = [enemy2, enemy1]

        if self.choice == [False, False]:
            if self.index < self.teammate:
                enemy = enemylist[0]
                enemy_teammate = enemylist[1]
            else:
                enemy = enemylist[1]
                enemy_teammate = enemylist[0]
        else:
            enemy = enemylist[0]
        if self.gridSafeMap[enemy[0]] * 2 + 1 >= self.getMazeDistance(gameState.getAgentPosition(self.index), enemy[0]) and not self.teamIsPanman(gameState, self.nearestLive(gameState, enemy[0])):
            goal = [self.nearestLive(gameState, enemy[0])]
        else:
            minDist1 = min([self.getMazeDistance(food, enemy[0]) for food in self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)])
            goal1 = [self.nearestLive(gameState, food) for food in self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState) if self.getMazeDistance(food, enemy[0]) == minDist1]
            goal = goal1
            if self.choice == [False, False]:
                minDist2 = min([self.getMazeDistance(food, enemy_teammate[0]) for food in self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)])
                goal2 = [self.nearestLive(gameState, food) for food in self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState) if self.getMazeDistance(food, enemy_teammate[0]) == minDist2]
                problem1 = PacmanContestProblem(self, gameState, goal1)
                state1, action1 = aStarSearch2(self, gameState, problem1, pacmanContestHeuristic)
                problem2 = PacmanContestProblem(self, gameState, goal2)
                state2, action2 = aStarSearch2(self, gameState, problem2, pacmanContestHeuristic)
                if state1 == state2:
                    return Directions.STOP
                elif self.getMazeDistance(gameState.getAgentPosition(self.index), state1) + self.getMazeDistance(gameState.getAgentPosition(self.teammate), state2) <= self.getMazeDistance(gameState.getAgentPosition(self.index), state2) + self.getMazeDistance(gameState.getAgentPosition(self.teammate), state1):
                    goal = goal1
                    action = action1
                else:
                    goal = goal2
                    action = action2

                self.debugDraw(goal, [110, 110, 110], False)

                if action:
                    return action[0]
                else:
                    return Directions.STOP

        self.debugDraw(goal, [110, 110, 110], False)

        problem = PacmanContestProblem(self, gameState, goal)
        state, action = aStarSearch2(self, gameState, problem, pacmanContestHeuristic)
        if action:
            return action[0]
        else:
            return Directions.STOP


class PacmanContestProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, agent, gameState, goal, start=None, costFn=lambda x: 1, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.agent = agent
        self.goal = goal
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        self.home = agent.getHome(gameState)
        self.walls = gameState.getWalls()
        self.startState = gameState.getAgentPosition(agent.index)
        if start != None: self.startState = start
        self.costFn = lambda x: 1
        if costFn != None: self.costFn = costFn
        self.visualize = visualize

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state in self.goal
        """
        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable
        """
        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


def nullHeuristic2(state, gameState, nextState):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def pacmanContestHeuristic(agent, gameState, nextState):
    return -1 * agent.getMazeDistance(gameState.getAgentPosition(agent.index),
                                      gameState.getAgentPosition(agent.teammate))


def breadthFirstSearch(problem):
    from util import Queue
    queue = Queue()
    closedList = []
    actions = []
    queue.push((problem.getStartState(), actions))
    while not queue.isEmpty():
        state, actions = queue.pop()
        if problem.isGoalState(state):
            return state, actions
        else:
            if state not in closedList:
                successor = problem.getSuccessors(state)
                closedList.append(state)
                for nextState, action, cost in successor:
                    queue.push((nextState, actions + [action]))


def aStarSearch2(agent, gameState, problem, heuristic=nullHeuristic2):
    from util import PriorityQueue
    queue = PriorityQueue()
    closedList = []
    actions = []
    queue.push((problem.getStartState(), actions), 0)
    while not queue.isEmpty():
        state, actions = queue.pop()
        if problem.isGoalState(state):
            return state, actions
        else:
            if state not in closedList:
                successor = problem.getSuccessors(state)
                closedList.append(state)
                for nextState, action, cost in successor:
                    queue.push((nextState, actions + [action]),
                               problem.getCostOfActions(actions + [action]) + heuristic(agent, gameState, nextState))
