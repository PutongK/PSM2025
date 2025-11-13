//
//  main.cpp
//  Elasticity
//
//  Created by Wim van Rees on 2/15/16. 
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#include "common.hpp"
#include "ArgumentParser.hpp"
#include "Sim.hpp"

#include "Sim_Bilayer_Growth.hpp"
#include "Sim_InverseBilayer.hpp"
#include "Sim_Calibration.hpp"

#ifdef USETBB
#include "tbb/task_scheduler_init.h"
#endif

int main(int argc,  const char ** argv)
{
#ifdef USETBB
    const int num_threads = tbb::task_scheduler_init::default_num_threads();
    tbb::task_scheduler_init init(num_threads);
    std::cout << "Starting TBB with " << num_threads << " threads " << std::endl;
#endif

    BaseSim * sim = nullptr;

    ArgumentParser parser(argc,argv);

    const std::string simCase = parser.parse<std::string>("-sim", "");

    if(simCase == "bilayer_growth")
        sim = new Sim_Bilayer_Growth(parser);
    else if(simCase == "inverse_bilayer")
        sim = new Sim_InverseBilayer(parser);
    else if(simCase == "calibration")
        sim = new Sim_Calibration(parser);

    else
    {
        std::cout << "No valid sim case defined. Options are \n";

        std::cout << "\t -sim bilayer_growth\n";
        std::cout << "\t -sim inverse_bilayer\n";
        std::cout << "\t -sim calibration\n";

        helpers::catastrophe("sim case does not exist",__FILE__,__LINE__);
    }

    sim->init();
    sim->run();

    delete sim;

    parser.finalize();

    return 0;
}
