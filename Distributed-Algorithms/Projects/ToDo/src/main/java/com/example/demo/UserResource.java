package com.example.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import javax.ws.rs.Consumes;
import javax.ws.rs.core.MediaType;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class UserResource {
	private UserRepository repo = new UserRepository();
	
	@GetMapping("api/users")
	public List<User> all(@RequestHeader Map<String, String> headers){
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		return repo.getAll(data.get(0), data.get(1));
	}
	
	@GetMapping("api/users/{id}")
	public User one(@PathVariable Long id, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		return repo.getById(id, data.get(0), data.get(1));
	}
	
	@PostMapping("api/users/create")
	public User create(@RequestBody User user) {
		if(repo.add(user)) {
			return user;
		}
		return new User();
	}
	
	@PutMapping("api/users/update/{id}")
	@Consumes(MediaType.APPLICATION_JSON)
	public User update(@PathVariable Long id, @RequestBody User user) {
		return repo.update(id, user);
	}
	
	@DeleteMapping("api/users/remove/{id}")
	public void remove(@PathVariable Long id, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		repo.remove(id, data.get(0), data.get(1));
	}
}
