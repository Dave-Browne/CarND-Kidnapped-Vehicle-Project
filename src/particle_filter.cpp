/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

    // Variable declarations
    Particle P;                          // single particle
    num_particles = 1;                  // number of particles
    particles.resize(num_particles);
    weights.resize(num_particles);

    // Gaussian noise
    default_random_engine gen;           // generator for gaussian distribution
    normal_distribution<double> x_gauss(x, std[0]);
    normal_distribution<double> y_gauss(y, std[1]);
    normal_distribution<double> theta_gauss(theta, std[2]);

    // Define particle coords according to a gps location + gaussian noise
    for (int i=0; i<num_particles; ++i) {
        P.x = x_gauss(gen);
        P.y = y_gauss(gen);
        P.theta = theta_gauss(gen);
        P.weight = 1;                                // initialize weights to 1
        particles[i] = P;
        weights[i] = P.weight;
    }
    is_initialized = true;
    cout << "Initialization complete, " << num_particles << " particles: x = " << P.x << ", y = " << P.y << ", theta = " << P.theta << endl;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Gaussian noise
    default_random_engine gen;           // generator for gaussian distribution
    normal_distribution<double> x_gauss(0, std_pos[0]);
    normal_distribution<double> y_gauss(0, std_pos[1]);
    normal_distribution<double> theta_gauss(0, std_pos[2]);

    // Loop over each particle and update it's x, y and theta values
    for (auto& P : particles) {                      // & defines the pointer which allows particles to be updated as P changes

        if (yaw_rate == 0) {
            P.x += velocity * delta_t * cos(P.theta);
            P.y += velocity * delta_t * sin(P.theta);
        }
        else {
            P.x += velocity / yaw_rate * ( sin(P.theta+yaw_rate*delta_t) - sin(P.theta) );
            P.y += velocity / yaw_rate * ( cos(P.theta) - cos(P.theta+yaw_rate*delta_t) );
            P.theta += yaw_rate * delta_t;
        }
        P.x += x_gauss(gen);
        P.y += y_gauss(gen);
        P.theta += theta_gauss(gen);
    }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // predicted is a vector of the landmarks in range of the car (map coords)
    // observations is a vector of observed coords from the car to be applied to each particle (map coords)
    //   - if the observed coord from a particle == a landmark, the particle is in the position of the car.

    // Variable declarations
    double distance, min_dist;

    // Find the distance from each observation to it's closest landmark
    for (auto& obs : observations) {
        min_dist = 999999;

        // Check each landmark distance
        for (size_t j=0; j<predicted.size(); ++j) {
            distance = dist(obs.x, obs.y, predicted[j].x, predicted[j].y);
            if (distance < min_dist) {
                min_dist = distance;
                obs.id = j;
            }
        }
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    // Variable declarations
    LandmarkObs lm;                                // single landmark
    vector<LandmarkObs> transformed_observations;  // vector of observations transformed to the map coordinate system
    vector<LandmarkObs> landmarks_in_range;        // vector of landmarks in range of the car
    double distance;
    double sigma_x, sigma_y, x, y, mu_x, mu_y, exponent, MGP;

    // Clear particle's weights
    weights.clear();

    // Identify the landmarks in range of the car
    for (auto& landmark : map_landmarks.landmark_list) {
        distance = dist(0, 0, landmark.x_f, landmark.y_f);
        if (distance <= sensor_range) {
            lm.x = landmark.x_f;
            lm.y = landmark.y_f;
            lm.id = landmark.id_i;
            landmarks_in_range.push_back(lm);
        }
    }

    // Transform each particle's observations from vehicle to map coordinate system
    for (auto& P : particles) {
        // Reset weight to 1 so that the new weight can be calculated
        P.weight = 1;
        // Clear transformed_observations so that observations for a new particle can be pushed in
        transformed_observations.clear();
        // Transform this particle's observations...
        // Be careful: overwriting obs.x will affect the transformed y observation! That is why lm is re-used
        for (auto obs : observations) {
            lm.x = P.x + (obs.x * cos(P.theta)) - (obs.y * sin(P.theta));
            lm.y = P.y + (obs.x * sin(P.theta)) + (obs.y * cos(P.theta));
            transformed_observations.push_back(lm);
        }
        // Now that this particle's landmark observations have been transformed,
        //   we need to find it's associated landmark ID using the dataAssociation function.
        //   After that we can update it's weight and move onto the next particle
        dataAssociation(landmarks_in_range, transformed_observations);

        // Calculate Multivariate Gaussian Probability of each transformed observation (product of these = weight)
        // transformed_observations contains the ID of landmarks_in_range...
        sigma_x = std_landmark[0];
        sigma_y = std_landmark[1];
        for (auto t_obs : transformed_observations) {
            x = t_obs.x;
            y = t_obs.y;
            mu_x = landmarks_in_range[t_obs.id].x;
            mu_y = landmarks_in_range[t_obs.id].y;
            exponent = -((pow((x-mu_x), 2)/(2.*pow(sigma_x, 2))) + (pow((y-mu_y), 2)/(2.*pow(sigma_y, 2))));
            MGP = (1 / (2. * M_PI * sigma_x * sigma_y)) * exp(exponent);
            P.weight *= MGP;
        }
        weights.push_back(P.weight);
    }
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Declare variables
    std::vector<Particle> new_particles;
    new_particles.resize(num_particles);
    new_particles.clear();

    // Use discrete_distribution to obtain new weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for(int i=0; i<num_particles; ++i) {
        new_particles.push_back(particles[d(gen)]);
    }
//    // TESTING
//    for (int i=0; i<num_particles; ++i) {
//        cout << "PRE : " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << endl;
//    } // END OF TEST
    particles = new_particles;
//    // TESTING
    for (int i=0; i<num_particles; ++i) {
        cout << "POST: " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << " " << particles[i].weight << endl;
    } // END OF TEST
}


void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
