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
  num_particles = 1000;
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
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine rand_eng;
  normal_distribution<double> dist_velocity(velocity, std_pos[0]);
  normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[2]);

  for (unsigned int i = 0; i < this->num_particles; i++) {
    double vel = dist_velocity(rand_eng);
    double yrate = dist_yaw_rate(yaw_rate);
    double new_theta = particles[i].theta + yrate * delta_t;
    double new_x = particles[i].x + vel / yrate * (sin(new_theta) - sin(particles[i].theta));
    double new_y = particles[i].y + vel / yrate * (cos(particles[i].theta) - cos(new_theta));

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
  for (unsigned int i = 0; i < predicted.size(); i++) {
    LandmarkObs *roi = &predicted[i];
    double min_dist = 0;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double dist = pow((roi->x - observations[i].x), 2) + pow((roi->y - observations[i].y), 2);
      if (j == 0) {
        observations[j].id = roi->id;
        min_dist = dist;
      } else {
        if (dist < min_dist) {
          observations[j].id = roi->id;
          min_dist = dist;
        } // if (dist < min_dist)
      } // if (i == 0)
    } // for (unsigned int j = 0; j < observations.size(); j++)
  } // for (unsigned int i = 0; i < predicted.size(); i++)
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
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
