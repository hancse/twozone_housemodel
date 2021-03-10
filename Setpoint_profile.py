'''
Setpoint_profile: Create temperature SP profile

'''


import numpy as np

class SP_T:
    """ Making an temperature setpoint profile
		 
		 Args:
		 
			(scalar) days_hours 	: number_of_hour_in_oneday + start hour at 0
			(scalar) days 			: number of simulation days (365 days for a year)
			(scalar) Night_T_SP  	: temperature SP at night time
			(scalar) Day_T_SP  		: temperature SP at dat time/ working hours.
			
			
		 Returns:
		 
			SP	: Temperature Set point
			
    """
    
    
    def __init__(self, days_hours,days,Night_T_SP,Day_T_SP):
	
        self.days_hours = days_hours
        self.days = days
        self.Night_T_SP = gamma
        self.Day_T_SP = Day_T_SP
        

    def week_day_profile(self):
	
		#temperature different between day and night.
		delta_T= Day_T_SP - Night_T_SP

		# Define Wake up time
		Wu_time =7           # wake up at 7 in the morning
		duty_wu = 23-7       

		# Go to work time/ leave the house
		Work_time = 8           #go to work at 8 in the morning
		duty_w   = 23-8        

		# Back to home
		back_home = 18         #back home at 18.00
		duty_b   = 23-18 


		#create simulation time
		time_t = np.linspace(0,periods,(days_hours*days)+1)

		#-----------------------
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp1 = signal.square(2 * np.pi* days * t,duty=duty_wu/24)
		temp1 = np.clip(temp1, 0, 1)
		# add delay to array
		temp1=np.roll(temp1,Wu_time)

		#----------------
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp2 = signal.square(2 * np.pi* days * t,duty=duty_w/24)
		temp2 = np.clip(temp2, 0, 1)
		# add delay to array
		temp2=np.roll(temp2,Work_time)

		#___________
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp3 = signal.square(2 * np.pi* days * t,duty=duty_b/24)
		temp3 = np.clip(temp3, 0, 1)
		# add delay to array
		temp3=np.roll(temp3,back_home)

		# Calculate SP
		temp4=temp1-temp2+temp3
		SP=(temp4*delta_T)+Night_T_SP

		SP=SP[np.newaxis]
		SP=SP.T
		SP=np.delete(SP, -1, 0)
		
	return SP
	
	def day_off_profile(self):
		
		
		"""Temperature different between day and night."""
		
		delta_T= Day_T_SP - Night_T_SP

		"""Define wake up time """
		W_time  =10;           """wake up at 10: during day-off """
		duty_wo = 23-10       

		"""Heating system turn off at 22:00"""
		S_time    = 22;           """Heating system turn off at 22:00 """
		duty_ho   = 23-22        


		#-----------------------
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp1 = signal.square(2 * np.pi* days * t,duty=duty_wo/24)
		temp1 = np.clip(temp1, 0, 1)
		# add delay to array
		temp1=np.roll(temp1,W_time)

		#----------------
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp2 = signal.square(2 * np.pi* days * t,duty=duty_ho/24)
		temp2 = np.clip(temp2, 0, 1)
		# add delay to array
		temp2=np.roll(temp2,S_time)

		'''
		#___________
		t= np.linspace(0,1,(days_hours*days)+1,endpoint=False)          #+1 start from 0 days=1
		temp3 = signal.square(2 * np.pi* days * t,duty=duty_b/24)
		temp3 = np.clip(temp3, 0, 1)
		# add delay to array
		temp3=np.roll(temp3,back_home)
		'''

		# Calculate SP

		temp4=temp1-temp2#+temp3
		SP=(temp4*delta_T)+Night_T_SP

		SP=SP[np.newaxis]
		SP=SP.T
		SP=np.delete(SP, -1, 0)
	
	return SP
	