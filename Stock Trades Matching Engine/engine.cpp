#include <iostream>
#include <thread>

#include "io.hpp"
#include "engine.hpp"

void Engine::accept(ClientConnection connection)
{
	auto thread = std::thread(&Engine::connection_thread, this, std::move(connection));
	thread.detach();
}

void Engine::connection_thread(ClientConnection connection)
{
	while(true)
	{
		ClientCommand input {};
		switch(connection.readInput(input))
		{
			case ReadResult::Error: SyncCerr {} << "Error reading input" << std::endl;
			case ReadResult::EndOfFile: return;
			case ReadResult::Success: break;
		}

		// Functions for printing output actions in the prescribed format are
		// provided in the Output class:
		switch(input.type)
		{
			case input_cancel: {
				// Remember to take timestamp at the appropriate time, or compute
				// an appropriate timestamp!
				bool found = false;
					
				// check buylist 
				found = buyList.cancel(input.order_id);
				if (found) {
					break;
				}
		
				// check sellList 
				found = sellList.cancel(input.order_id);
				if (found) {
					break;
				}
				
				if (!found) {
					auto output_time = getCurrentTimestamp();
					Output::OrderDeleted(input.order_id, false, output_time);
				}
				break;
			}

			case input_buy: {
				// delays until the prev sell consecutive orders finishes
				while (sellMatchingCount.load(std::memory_order_acquire) != 0) {}	

				// first buyer
				if (buyMatchingCount.fetch_add(1) == 0) 
				{
					uint32_t remaining_buy_count = sellList.match(input.order_id, input.price, input.count, input.instrument);

					// buy order not fully filled
					if (remaining_buy_count) {
						buyList.addToList(input.instrument, input.order_id, input.price, remaining_buy_count);
					}

					while (buyMatchingCount.load() != 1) {}
					buyMatchingCount.fetch_sub(1, std::memory_order_release);
				} 
				else 
				{
					uint32_t remaining_buy_count = sellList.match(input.order_id, input.price, input.count, input.instrument);

					// buy order not fully filled
					if (remaining_buy_count) {
						buyList.addToList(input.instrument, input.order_id, input.price, remaining_buy_count);	
					}
					buyMatchingCount.fetch_sub(1);
				}
				break;
			}
			case input_sell: {
				// delays until the prev buy consecutive orders finishes
				while (buyMatchingCount.load(std::memory_order_acquire) != 0) {}
				
				// first seller
				if (sellMatchingCount.fetch_add(1) == 0) 
				{
					uint32_t remaining_sell_count = buyList.match(input.order_id, input.price, input.count, input.instrument);

					// sell order not fully filled
					if (remaining_sell_count) {
						sellList.addToList(input.instrument, input.order_id, input.price, remaining_sell_count);
					}
					
					while (sellMatchingCount.load() != 1) {}
					sellMatchingCount.fetch_sub(1, std::memory_order_release);
				}
				else 
				{	
					uint32_t remaining_sell_count = buyList.match(input.order_id, input.price, input.count, input.instrument);

					// sell order not fully filled
					if (remaining_sell_count) {
						sellList.addToList(input.instrument, input.order_id, input.price, remaining_sell_count);
					} 

					sellMatchingCount.fetch_sub(1);
				}
				
				break;
			}
			default: {
				// SyncCerr {}
				//     << "Got order: " << static_cast<char>(input.type) << " " << input.instrument << " x " << input.count << " @ "
				//     << input.price << " ID: " << input.order_id << std::endl;

				// Remember to take timestamp at the appropriate time, or compute
				// an appropriate timestamp!
				auto output_time = getCurrentTimestamp();
				Output::OrderAdded(input.order_id, input.instrument, input.price, input.count, input.type == input_sell,
				    output_time);
				break;
			}
		}
		
		// Additionally:

		// Remember to take timestamp at the appropriate time, or compute
		// an appropriate timestamp!
		//intmax_t output_time = getCurrentTimestamp();

		// Check the parameter names in `io.hpp`.
		//Output::OrderExecuted(123, 124, 1, 2000, 10, output_time);
	}
	
}
