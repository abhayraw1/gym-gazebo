import gym
import tf
import yaml
import rospy
import roslaunch
import time
import numpy as np

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates 
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Odometry
import turtlesim.msg


from gym.utils import seeding
import sys, os
import subprocess



TB  = "/turtlebot_{}"
RSF = 20 # Reward Scaling Factor

class GazeboFormationTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Read configuration file
        self.configure()
        # Launch the simulation with the given launchfile name
        super(GazeboFormationTurtlebotLidarEnv, self).__init__("GazeboFormationTurtlebotsLidar_v0.launch")
        # Launch turtlebots using configuration
        self.launch_turtles()
        self.previous_actions = {TB[1:].format(i):[0., 0.] for i in xrange(self.config['num_agents'])}
        self.tb_pairs = [(TB[1:].format(i), TB[1:].format(j)) for i in range(self.config['num_agents'])\
                         for j in range(self.config['num_agents']) if i != j]
        self.init_subscribers()
        self.vel_pub = [rospy.Publisher('/turtlebot_{}/mobile_base/commands/velocity'.format(i), Twist, queue_size=5)\
                        for i in range(self.config['num_agents'])]
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.timestamp = rospy.wait_for_message('/clock', Clock, timeout=5).clock
        # wait for setup

        rospy.wait_for_message('/tf', TFMessage, timeout=5)
        [rospy.wait_for_message(TB.format(i)+'/ground_truth/state'.format(i), Odometry, timeout=10) \
            for i in range(self.config['num_agents'])]
        self.tf_listener = tf.TransformListener(True, rospy.Duration(10.0))
        while True:
            c = True
            for i in range(self.config['num_agents']):
                print TB[1:].format(i)+"/base_link", self.tf_listener.frameExists(TB[1:].format(i)+"/base_link"), c
                print self.tf_listener.getFrameStrings()
                c = self.tf_listener.frameExists(TB[1:].format(i)+"/base_link") and c
            if c: break

        self.reward_range = (-np.inf, np.inf)
        self._seed()
        self.get_state_value(self.get_state_data())


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def updateTimestamp(self, msg):
        self.timestamp = msg.clock

    def handle_turtle_pose(self, msg):
        tbot_arr = [j for j in msg.name if "turtlebot" in j]
        for i in tbot_arr:
            idx = msg.name.index(i)
            pos = msg.pose[idx].position
            ori = msg.pose[idx].orientation
            orientation = [ori.x, ori.y, ori.z, ori.w]
            position = [pos.x, pos.y, pos.z]
            br = tf.TransformBroadcaster()
            br.sendTransform(position, orientation, self.timestamp, i+"/base_link", "/world")
            br.sendTransform(position, orientation, self.timestamp, i+"/odom", "/world")

    def init_subscribers(self):
        rospy.Subscriber('/clock', Clock, self.updateTimestamp)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.handle_turtle_pose)

    def launch_turtles(self):
        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        launch_file = "GazeboTurtlebotLidar.launch"
        tb_launch_file = os.path.join(os.path.dirname(__file__), "..",  "assets", "launch", launch_file)
        for i in range(self.config['num_agents']):
            subprocess.Popen([sys.executable, os.path.join(ros_path, b"roslaunch"), \
                              "-p", self.port, tb_launch_file, "robotID:={}".format(i), \
                              "x:={}".format(i), "y:={}".format(i)])

    def configure(self):
        current_path = os.path.abspath(os.path.dirname(__file__))
        config_file_path = "/../assets/config/formation.yaml"
        try:
            config_file = open(current_path + config_file_path, "r")
            self.config = yaml.load(config_file)
            self.generate_formation()
        except Exception as e:
            print (e)
            print ("Failed to load file: {}".format(config_file_path))
            print ("Exiting...")
            self._close()

    def generate_formation(self):
        x = {}
        for i, val_i in self.config['formation'].items():
            for j, val_j in self.config['formation'].items():
                if i == j: continue
                x[TB[1:].format(i), TB[1:].format(j)] = [b - a for a,b in zip(val_i, val_j)]
        self.config['formation'] = x
        print x

    def get_state_data(self):
        state_data = {i:{} for i in xrange(self.config['num_agents'])}
        for source_tb, target_tb in self.tb_pairs:
            source_frame = source_tb + "/base_link"
            target_frame = target_tb + "/base_link"
            if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):
                common_time = self.tf_listener.getLatestCommonTime(source_frame, target_frame)
                source2target = self.tf_listener.lookupTransform(target_frame, source_frame, common_time)
                state = source2target[0][:2] + [tf.transformations.euler_from_quaternion(source2target[1])[-1]]
                state += self.config['formation'][target_tb, source_tb]
                state += self.previous_actions[target_tb] + self.previous_actions[source_tb]
                state_data[int(target_tb.split("_")[-1])][int(source_tb.split("_")[-1])] = state
        print state_data
        return state_data

    def get_reward(self, data):
        reward = self.prev_state_value
        reward = self.get_state_value(data) - reward
        print "REWARD: ", reward*RSF
        return reward*RSF

    def get_state_value(self, data):
        # squared sum of all the errors in distances
        # value is invese unfavourability of the state theerfore negative
        v1, v2 = 0, 0
        states = [i[j] for i in data.values() for j in i.keys()]
        for state in states:
            v1 -= (state[0] - state[3])**2
            v1 -= (state[1] - state[4])**2
            v2 -= state[2]**2
        try:
            print "PREVIOUS VALUE: {}".format(self.prev_state_value)
            self.prev_state_value = 0
        except:
            print "INITIAL STATE"
        self.prev_state_value = (v1 + v2)
        print "CURRENT VALUE: {}".format(self.prev_state_value)
        return self.prev_state_value

    def _step(self, actions):
        assert type(actions) == dict, "Provide actions as a dict {<tbot_id> : <action>}"
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        vel_cmd = Twist()
        for i in xrange(self.config['num_agents']):
            vel_cmd.linear.x, vel_cmd.angular.z = actions[i]
            self.vel_pub[i].publish(vel_cmd)


        ###############
        ##   TODOS   ###
        #################
        ###############################################################################
        # 1. get the relative positions and orientations for all pairs of turtlebots. # Done
        # 2. containr for previous command velocities for every turtlebot.            # Done
        # 3. get formation from the configuration file.                               # Done
        # 4. convert the formation into relative positions and orientations.          # Done
        # 5. form a state vector for each of the turtlebots.                          # Done
        # 6. reward function                                                          # Done
        # 7. reset function                                                           # 
        # 8. random state intializer                                                  #
        # 9. random goal intializer                                                   #
        ###############################################################################

        obs = self.get_state_data()
        for i in actions.keys():
            self.previous_actions[TB[1:].format(i)] = actions[i]
        reward = self.get_reward(obs)
        
