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
import turtlesim.msg


from gym.utils import seeding
import sys, os
import subprocess



TB = "/turtlebot_{}"
class GazeboFormationTurtlebotLidarEnv(gazebo_env.GazeboEnv):

    def __init__(self):
        # Read configuration file
        self.configure()
        # Launch the simulation with the given launchfile name
        super(GazeboFormationTurtlebotLidarEnv, self).__init__("GazeboFormationTurtlebotsLidar_v0.launch")
        # Launch turtlebots using configuration
        self.launch_turtles()
        self.tb_pairs = [(TB[1:].format(i), TB[1:].format(j)) for i in range(self.config['num_agents'])\
                         for j in range(self.config['num_agents']) if i != j]
        self.init_subscribers()
        self.vel_pub = [rospy.Publisher('/turtlebot_{}/mobile_base/commands/velocity'.format(i), Twist, queue_size=5)\
                        for i in range(self.config['num_agents'])]
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.timestamp = rospy.wait_for_message('/clock', Clock, timeout=5).clock
        t = 0
        while t < 10:
            rospy.wait_for_message('/tf', TFMessage, timeout=5)
            t +=2
        self.reward_range = (-np.inf, np.inf)
        self.tf_listener = tf.TransformListener(True, rospy.Duration(10.0))

        self._seed()
        self.get_state_data()


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
            br.sendTransform(position, orientation, self.timestamp, "/"+i+"/odom", "/world")

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
        except Exception as e:
            print (e)
            print ("Failed to load file: {}".format(config_file_path))
            print ("Exiting...")
            self._close()


    def get_state_data(self):
        state_data = {}
        for i, j in self.tb_pairs:
            source_frame = j+"/base_footprint"
            target_frame = i+"/base_footprint"
            if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):
                print i, j
                common_time = self.tf_listener.getLatestCommonTime(source_frame, target_frame)
                j_to_i = self.tf_listener.lookupTransform(i+"/base_footprint", j+"/base_footprint", common_time)
                print j_to_i


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
        # 1. get the relative positions and orientations for all pairs of turtlebots. #
        # 2. containr for previous command velocities for every turtlebot.            #
        # 3. get formation from the configuration file.                               #
        # 4. convert the formation into relative positions and orientations.          #
        # 5. form a state vector for each of the turtlebots.                          #
        # 6. reward function                                                          #
        # 7. reset function                                                           #
        # 8. random state intializer                                                  #
        # 9. random goal intializer                                                   #
        ###############################################################################

        self.get_state_data()