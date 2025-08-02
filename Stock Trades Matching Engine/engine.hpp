// This file contains declarations for the main Engine class. You will
// need to add declarations to this file as you develop your Engine.

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <chrono>
#include <unordered_map>
#include <string>
#include <mutex>
#include <memory>
#include <atomic>
#include <condition_variable>

#include "io.hpp"

inline std::chrono::microseconds::rep getCurrentTimestamp() noexcept
{
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct instrumentNode 
{	
public:
	uint32_t order_id;
	uint32_t price;
	uint32_t count;
	bool isLastNode;
	std::atomic<uint32_t> exe_id{1};
	std::mutex mut_node;
	std::string instrument;
	instrumentNode* next;
	
	instrumentNode(uint32_t order_id, uint32_t price, uint32_t count, bool isLastNode, std::string instrument)
		: order_id(order_id), price(price), count(count), isLastNode(isLastNode), instrument(instrument), next(nullptr) {}
};

struct instrumentList 
{
public:
	instrumentList(bool isAsc) : isAsc(isAsc) {
		if (isAsc) {
			head = new instrumentNode(0, UINT32_MAX, 0, true, "dummy");
		} else {
			head = new instrumentNode(0, 0, 0, true, "dummy");
		}
	}

	// sorted in desc for buy, incr for sell
	void addToList(char* instrument, uint32_t order_id, uint32_t price, uint32_t count) {
		instrumentNode *new_node = new instrumentNode(order_id, price, count, false, instrument);
		std::unique_lock<std::mutex> lock_head(head_mut);
		instrumentNode *curr = head;
		std::unique_lock<std::mutex> lock_curr(curr->mut_node);
		std::unique_lock<std::mutex> lock_prev;
		instrumentNode *prev = nullptr;
		std::atomic<bool> unlockHead{true};

		if (isAsc) {
			while (curr and !curr->isLastNode) {
				if (price < curr->price) break;

				if (unlockHead.load()) {
					lock_head.unlock();
					unlockHead.store(false);
				}
				
				std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
				lock_prev = std::move(lock_curr);
				prev = curr;
				curr = curr->next;

				lock_curr = std::move(lock_next);
			}
		} else {
			while (curr and !curr->isLastNode) {
				if (price > curr->price) break;

				if (unlockHead.load()) {
					lock_head.unlock();
					unlockHead.store(false);
				}

				std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
				lock_prev = std::move(lock_curr);
				prev = curr;
				curr = curr->next;

				lock_curr = std::move(lock_next);
			}
		}

		// checks if prev is not null
		if (prev) {
			prev->next = new_node;
			new_node->next = curr;
		} else {
			new_node->next = curr;
			head = new_node;
		}

		auto output_time = getCurrentTimestamp();
		Output::OrderAdded(order_id, instrument, price, count, isAsc, output_time);
	}

	uint32_t match(uint32_t order_id, uint32_t price, uint32_t count, std::string instrument) {
		std::unique_lock<std::mutex> lock_head(head_mut);
		instrumentNode *curr = head;
		std::unique_lock<std::mutex> lock_curr(curr->mut_node);
		std::atomic<bool> unlockHead{true};

		if (isAsc) {
			// buy order matching resting sell orders
			while (curr and !curr->isLastNode and count) {
				if (unlockHead.load()) {
					lock_head.unlock();
					unlockHead.store(false);
				}

				if (instrument != curr->instrument) {
					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
					continue;
				}

				if (price < curr->price) break;

				if (!curr->count) {
					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
					continue;
				}

				if (curr->count >= count) {
					// all filled
					curr->count -= count;
					auto output_time = getCurrentTimestamp();

					Output::OrderExecuted(curr->order_id, order_id, curr->exe_id.load(), curr->price, count, output_time);;
					curr->exe_id.fetch_add(1);
					count = 0;
					break;
				} else {
					// curr buy order partially filled
					count -= curr->count;
					auto output_time = getCurrentTimestamp();
					Output::OrderExecuted(curr->order_id, order_id, curr->exe_id.load(), curr->price, curr->count, output_time);;
					curr->count = 0;

					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
				}	
			}

		} else {
			// sell order matching resting buy orders
			while (curr and !curr->isLastNode and count) {
				if (unlockHead.load()) {
					lock_head.unlock();
					unlockHead.store(false);
				}

				if (instrument != curr->instrument) {
					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
					continue;
				}

				if (price > curr->price) break;

				if (!curr->count) {
					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
					continue;
				}

				if (curr->count >= count) {
					// all filled
					curr->count -= count;
					auto output_time = getCurrentTimestamp();

					Output::OrderExecuted(curr->order_id, order_id, curr->exe_id.load(), curr->price, count, output_time);;
					curr->exe_id.fetch_add(1);
					count = 0;
					break;
				} else {
					count -= curr->count;
					auto output_time = getCurrentTimestamp();
					Output::OrderExecuted(curr->order_id, order_id, curr->exe_id.load(), curr->price, curr->count, output_time);;
					curr->count = 0;

					std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
					curr = curr->next;
					lock_curr.unlock();
					lock_curr = std::move(lock_next);
				}
			}
		}

		// return quantity left
		return count;
	}

	bool cancel(uint32_t order_id) {
		std::unique_lock<std::mutex> lock_head(head_mut);
		instrumentNode *curr = head;
		std::unique_lock<std::mutex> lock_curr(curr->mut_node);
		std::atomic<bool> unlockHead{true};
		
		while (curr and !curr->isLastNode) {
			if (unlockHead.load()) {
				lock_head.unlock();
				unlockHead.store(false);
			}

			if (curr->order_id == order_id) {
				auto output_time = getCurrentTimestamp();
				Output::OrderDeleted(order_id, curr->count != 0, output_time);
				curr->count = 0;
				return true;
			} else {
				std::unique_lock<std::mutex> lock_next(curr->next->mut_node);
				curr = curr->next;
				lock_curr.unlock();
				lock_curr = std::move(lock_next);
			}
		}
		return false;
	}

private:
	bool isAsc;
	instrumentNode* head;
	std::mutex head_mut;
};

struct Engine
{
public:
	void accept(ClientConnection conn);

private:
	instrumentList buyList{false};
	instrumentList sellList{true};

	//To handle multiple buys or multiple sells for each instrument
	std::atomic<int> sellMatchingCount{0};
	std::atomic<int> buyMatchingCount{0};

	void connection_thread(ClientConnection conn);
};

#endif
