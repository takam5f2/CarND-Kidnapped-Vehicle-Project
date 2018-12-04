/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // initilize number of particle.
  num_particles = 100;
  default_random_engine rand_eng;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[0]);
  normal_distribution<double> dist_theta(theta, std[0]);

  for (unsigned int i = 0; i < this->num_particles; i++) {
    Particle particle;
    particle.id = (int)i;
    particle.x = dist_x(rand_eng);
    particle.y = dist_y(rand_eng);
    particle.theta = dist_theta(rand_eng);
      
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine rand_eng;
  normal_distribution<double> dist_velocity(velocity, std_pos[0]);
  normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[2]);

  for (int i = 0; i < this->num_particles; i++) {
    double vel = dist_velocity(rand_eng);
    double yrate = dist_yaw_rate(rand_eng);
    double new_theta = particles[i].theta + yrate * delta_t;
    double new_x = particles[i].x + vel / (yrate + 1E-9) * (sin(new_theta) - sin(particles[i].theta));
    double new_y = particles[i].y + vel / (yrate + 1E-9) * (cos(particles[i].theta) - cos(new_theta));

    particles[i].x = new_x;
    particles[i].y = new_y;
    particles[i].theta = new_theta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
 	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (int i = 0; i < predicted.size(); i++) {
    LandmarkObs *roi = &predicted[i];
    double min_dist = 0;
    int min_idx = 0;
    for (int j = 0; j < observations.size(); j++) {
      double dist = pow((roi->x - observations[j].x), 2) + pow((roi->y - observations[j].y), 2);
      if (j == 0) {
        min_idx = j;
        min_dist = dist;
      } else {
        if (dist < min_dist) {
          min_idx = j;
          min_dist = dist;
        } // if (dist < min_dist)
      } // if (i == 0)
    } // for (int j = 0; j < observations.size(); j++)
    observations[min_idx].id = roi->id;
  } // for (int i = 0; i < predicted.size(); i++)
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  std::vector<LandmarkObs> roi_landmarks;
  std::vector<LandmarkObs> trans_observations;
  double weights_sum = 0;

  weights.clear();
  for (int i = 0; i < this->num_particles; i++) {
    roi_landmarks.clear();
    trans_observations.clear();
    Particle roi_particle = particles[i];
    double particle_weights = 1;
    // collect landmarks in sensor observating range.
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int land_id = map_landmarks.landmark_list[j].id_i;
      float land_x = map_landmarks.landmark_list[j].x_f;
      float land_y = map_landmarks.landmark_list[j].y_f;
      double land_dist = pow((roi_particle.x - land_x), 2) + pow((roi_particle.y - land_y), 2);
      land_dist = sqrt(land_dist);
      if (land_dist <= sensor_range) {
        LandmarkObs obs_landmark;
        obs_landmark.id = land_id;
        obs_landmark.x = land_x;
        obs_landmark.y = land_y;
        roi_landmarks.push_back(obs_landmark);
      }
    }
    // transform observation
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs trans_obs_landmark;
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      trans_obs_landmark.id = -1;
      trans_obs_landmark.x = roi_particle.x + (cos(roi_particle.theta) * obs_x) - (sin(roi_particle.theta) * obs_y);
      trans_obs_landmark.y = roi_particle.y + (sin(roi_particle.theta) * obs_x) + (cos(roi_particle.theta) * obs_y);
      trans_observations.push_back(trans_obs_landmark);
    }
    // association of observation and landmark
    this->dataAssociation(roi_landmarks, trans_observations);

    // update measurement probability
    for (int j = 0; j < trans_observations.size(); j++) {
      for (int k = 0; k < roi_landmarks.size(); k++) {
        if (trans_observations[j].id == roi_landmarks[k].id) {
          cout << i << " obs.id " << trans_observations[j].id << ", " << roi_landmarks[k].id << endl;
          cout << i << " obs.x " << trans_observations[j].x << ", " << roi_landmarks[k].x << endl;
          cout << i << " obs.y " << trans_observations[j].y << ", " << roi_landmarks[k].y << endl;
          double exp_x = (trans_observations[j].x - roi_landmarks[k].x);
          exp_x = exp_x * exp_x;
          exp_x = exp_x / (2*std_landmark[0]*std_landmark[0]);
          double exp_y = (trans_observations[j].y - roi_landmarks[k].y);
          exp_y = exp_y * exp_y;
          exp_y = exp_y / (2*std_landmark[1]*std_landmark[1]);
          double roi_weights = exp(-1 * (exp_x + exp_y));
          cout << i << " particle roi weights = " << roi_weights << ", sum = "<< (exp_x + exp_y) << endl;
          roi_weights = roi_weights / (2 * M_PI * std_landmark[0] * std_landmark[1]);
          particle_weights *= roi_weights;
        } // if 
      } // for (k = )
    } // for (j = )
    weights_sum += particle_weights;
    weights.push_back(particle_weights);
    cout << particle_weights << endl;
  }
  cout << "weights sum" << weights_sum << endl;
  // normalize particle weights
  for (int i = 0; i < weights.size(); i++) {
    weights[i] = weights[i] / weights_sum;
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine rand_eng;
  std::discrete_distribution<> d(weights.begin(), weights.end());

  std::vector<Particle> new_particles;

  for (int i = 0; i < num_particles; i++) {
    int idx = d(rand_eng);
    cout << "particle "<< idx << " is chosen which has weights of " << weights[idx] << endl;
    Particle new_particle;
    new_particle.id = particles[idx].id;
    new_particle.x = particles[idx].x;
    new_particle.y = particles[idx].y;
    new_particle.theta = particles[idx].theta;
    new_particle.weight = particles[idx].weight;
    new_particles.push_back(new_particle);
  }
  particles = new_particles;
  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
