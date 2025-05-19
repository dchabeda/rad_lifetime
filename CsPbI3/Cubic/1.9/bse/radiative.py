# Script that calculates the mean radiative rates and lifetimes from 
# quasiparticle and excitonic states
import numpy as np
import sys

def main():
    '''Main function for computing radiative lifetimes. Usage: python radiative.py'''
    # Set constants and prefactors
    temp = float(sys.argv[1])
    global AUTOS 
    global prefactor
    global kB
    AUTOS = 2.4188843265857e-17
    c = 137.03604
    eps0 = (1/(4*np.pi))

    prefactor = (3*np.pi) * eps0 * c**3 
    
    kB = 3.167e-6
    beta = 1/(kB*temp)
    cutoff = 3*kB*temp # set nstates cutoff using this.

    spinor_flag = 1
    if 'nospinor' in sys.argv:
        spinor_flag = 0
    calc_qp = 1
    if 'noqp' in sys.argv:
        calc_qp = 0
    which_states = ''
    if 'thermal' in sys.argv:
        which_states = 'thermal'
    if 'bright' in sys.argv:
        which_states = 'bright'
    
    global print_flag
    print_flag = 1
    if 'silent' in sys.argv:
        print_flag = 0

    if calc_qp:
        os_files = ['sorted_OS0.dat', 'OS.dat']
    else:
        os_files = ['OS.dat']

    if print_flag:
        print('\nRunning radiative.py...')

    for i in range(len(os_files)):
        if i == 0 and calc_qp == 1:
            flag = 'single particle'
            fflag = '0'
            str_fmt = 'Fundamental'
        else:
            flag = 'exciton'
            fflag = ''
            str_fmt = 'Optical'
        
        if print_flag:
            print('\n'+ '*'*43)
            print("Calculating {} radiative rates".format(flag))
            print('*'*43 + '\n')
        
        # Get the dipole moment values for each state and their corresponding energy
        os_file = np.loadtxt(os_files[i], dtype=np.float64)
        if fflag == '0':
            state = os_file[:,:2]
            ene = os_file[:,3]
            #state = np.concatenate((rs_file[:,0].reshape(len_rs,1),rs_file[:,1].reshape(len_rs,1)), axis=1)
        if fflag == '':
            state = os_file[:,0]
            ene = os_file[:,2]
        if spinor_flag:    
            os = os_file[:,-6:] # Grab the re and imag xyz coords
        else:
            os = os_file[:,-3:] # Grab the xyz coords

        # Calculate |mu|^2 for any general coords
        os2 = np.zeros(os.shape[0])
        for i in range(os.shape[1]):
            os2 += os[:,i]**2 
        #print('Oscillator strength: ', os2[:10])
        ene_cut, os2_cut = get_states(ene, os2, cutoff, which_states)
        rate, lifetime = calc_radiative(ene_cut, os2_cut, beta)
        if print_flag:
            print('Radiative rate = {:.4f} [ns^-1]'.format(rate) + '\nRadiative lifetime = {:.3f} [ns]\n'.format(lifetime) )
        # Save output file
        np.savetxt("avg_lifetimes.dat", np.array([[rate, lifetime]]), header='rates [ns^-1]    lifetimes[ns]', fmt="% .12f")
        analyze_output = np.array(np.hstack((ene_cut.reshape(-1,1), os2_cut.reshape(-1,1))))
        np.savetxt("energies_os.dat", analyze_output, fmt="%0.8f", header="E (a.u)    |mu|^2")

def calc_radiative(w, mu2, beta):
    
    Q = np.sum(np.exp(-w*beta))
    '''print("Q no diff: ", Q)
    ediff = w - w[0]
    print(ediff)
    Q = np.sum(np.exp(-ediff*beta))
    '''
    
    lifetimes = prefactor/(w**3 * mu2)*AUTOS*1e9
    rates = 1/lifetimes

    #np.savetxt("lifetimes.dat", np.array([[rates, lifetimes]]), header='rates [ns^-1]        lifetimes[ns]')

    avg_rate = 1/Q * np.sum(np.exp(-w*beta)* rates)
    avg_lifetime = 1/avg_rate

    return avg_rate, avg_lifetime


def get_states(ene, mu2, cutoff, which_states):
    # Exclude the lowest energy dark states
    dark_cut = 0.5
    if which_states == 'bright':
        if print_flag:
            print('Using only the lowest bright state')
            print('Dark state cutoff is |u|^2 = {}'.format(dark_cut))
        mindex = np.where(mu2 > dark_cut)[0][0]
        return ene[mindex], mu2[mindex]
    elif which_states == 'thermal':
        maxdex = np.where(ene > ene.min()+cutoff)[0][0]
        if print_flag:
            print('Using lowest 3kT states, kT = {:.2f} meV'.format(cutoff*27.211*1000))
            print('Using from state ', 0, ' to state ', maxdex)
        return ene[:maxdex], mu2[:maxdex]
    else:
        mindex = np.where(mu2 > dark_cut)[0][0]
        ene = ene[mindex:]
        maxdex = np.where(ene >= ene.min()+cutoff)[0][0]
        if print_flag:
            print('Using states 3kT above lowest bright state\n3kT = {:.2f} meV'.format(cutoff*27.211*1000))
            print('Using from state ', mindex, ' to state ', mindex + maxdex)
        return ene[:maxdex], mu2[mindex:mindex+maxdex]

main()