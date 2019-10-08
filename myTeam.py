# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
from util import nearestPoint
from util import PriorityQueue


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def getHome(self, gameState):
        homeList = []
        if self.red:
            x = gameState.data.layout.width // 2 - 1
        else:
            x = gameState.data.layout.width // 2
        for y in range(1, gameState.data.layout.height):
            if not gameState.hasWall(x, y):
                homeList.append((x, y))
        return homeList

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions if a != Directions.STOP]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        homeList = self.getHome(gameState)
        homeDistance = [self.getMazeDistance(successor.getAgentPosition(self.index), home) for home in homeList]
        features['homeDistance'] = successor.getAgentState(self.index).numCarrying / (min(homeDistance) + 1)

        features['getFood'] = successor.getAgentState(self.index).numCarrying

        foodList = self.getFood(successor).asList()
        foodDistance = [self.getMazeDistance(successor.getAgentPosition(self.index), food) for food in foodList]
        if foodDistance:
            features['foodDistance'] = 1 / (min(foodDistance) + 1)
        else:
            features['foodDistance'] = 0

        features['getCapsule'] = successor.getAgentPosition(self.index) in gameState.getCapsules()

        capsuleList = self.getCapsules(successor)
        capsuleDistance = [self.getMazeDistance(successor.getAgentPosition(self.index), capsule) for capsule in capsuleList]
        if capsuleDistance:
            features['capsuleDistance'] = 1 / (min(capsuleDistance) + 1)
        else:
            features['capsuleDistance'] = 0

        enemyList = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyDistance = [self.getMazeDistance(successor.getAgentPosition(self.index), enemy.getPosition()) for enemy in enemyList if not enemy.isPacman and enemy.getPosition() is not None]
        if enemyDistance and min(enemyDistance) < 5:
            features['enemyDistance'] = 1 / (min(enemyDistance) + 1)
        else:
            features['enemyDistance'] = 0

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 100, 'homeDistance': 1, 'getFood': 3, 'foodDistance': 1, 'getCapsule': 5, 'capsuleDistance': 2, 'enemyDistance': -10}


class DefensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def getHome(self, gameState):
        homeList = []
        if self.red:
            x = gameState.data.layout.width // 2 - 1
        else:
            x = gameState.data.layout.width // 2
        for y in range(1, gameState.data.layout.height):
            if not gameState.hasWall(x, y):
                homeList.append((x, y))
        return homeList

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions if a != Directions.STOP]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        homeList = self.getHome(gameState)
        homeDistance = self.getMazeDistance(successor.getAgentPosition(self.index), homeList[len(homeList) // 2])
        features['homeDistance'] = 1 / (homeDistance + 1)

        enemyList = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        enemyDistance = [self.getMazeDistance(successor.getAgentPosition(self.index), enemy.getPosition()) for enemy in
                         enemyList if enemy.isPacman and enemy.getPosition() is not None]
        if enemyDistance:
            features['enemyDistance'] = 1 / (min(enemyDistance) + 1)
        else:
            features['enemyDistance'] = 0

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'homeDistance': 1, 'enemyDistance': 10}